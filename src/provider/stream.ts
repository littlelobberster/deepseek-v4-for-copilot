import vscode from 'vscode';
import { createUserFacingError } from '../client';
import { getAutoContinueConfig } from '../config';
import { logger } from '../logger';
import type { CompletionSignal, DeepSeekRequest, DeepSeekToolCall, DeepSeekUsage } from '../types';
import {
	buildCompletionCheckRequest,
	buildContinuationRequest,
	decideContinuation,
	isCompletionCheckDone,
	looksUnfinished,
	type AutoContinueConfig,
} from './continuation';
import { DuplicateContentSuppressor } from './duplicate-content';
import {
	observeCancellationToken,
	type CacheDiagnosticsRun,
	type ReplayMarkerReportTrigger,
} from './debug';
import {
	createReplayMarkerPart,
	hasReplayMarkerMetadata,
	type ReplayMarkerMetadata,
} from './replay';
import type { PreparedChatRequest } from './request';
import { formatRequestLogLine, type RequestKind } from './routing';
import {
	createLeakedMarkerFilter,
	stripLeakedMarkers,
	stripTrailingMarkers,
	type LeakedMarkerFilter,
} from './leaked-markers';

interface ResponseStreamState {
	accumulatedReasoning: string;
	emittedToolCallIds: string[];
	initialResponseNoticeReported: boolean;
	replayMarkerReported: boolean;
}

const COPILOT_USAGE_DATA_PART_MIME = 'usage';

export interface StreamChatCompletionOptions {
	prepared: PreparedChatRequest;
	progress: vscode.Progress<vscode.LanguageModelResponsePart>;
	token: vscode.CancellationToken;
	initialResponseNotice?: string;
	getCharsPerToken: () => number;
	setCharsPerToken: (charsPerToken: number) => void;
}

export async function streamChatCompletion({
	prepared,
	progress,
	token,
	initialResponseNotice,
	getCharsPerToken,
	setCharsPerToken,
}: StreamChatCompletionOptions): Promise<void> {
	const state: ResponseStreamState = {
		accumulatedReasoning: '',
		emittedToolCallIds: [],
		initialResponseNoticeReported: false,
		replayMarkerReported: false,
	};
	const cancelListener = observeCancellationToken(token, prepared.cacheDiagnostics);
	const contentFilter = prepared.filterLeakedMarkers ? createLeakedMarkerFilter() : undefined;
	// The thinking channel already is reasoning: only bare tags are removed there, never text.
	const thinkingFilter = prepared.filterLeakedMarkers
		? createLeakedMarkerFilter({ dropReasoningBlocks: false })
		: undefined;
	// Gateways sometimes repeat the reasoning in the body with no tags; suppress that copy.
	const duplicateSuppressor = prepared.filterLeakedMarkers
		? new DuplicateContentSuppressor(() => state.accumulatedReasoning, prepared.isThinkingModel)
		: undefined;
	const autoContinue = getAutoContinueConfig();
	const canAutoContinue = autoContinue.enabled && prepared.requestKind === 'main-agent';

	let visibleText = '';

	const reportContent = (content: string): void => {
		if (!content) {
			return;
		}
		visibleText += content;
		reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
		progress.report(new vscode.LanguageModelTextPart(content));
	};

	const handleContent = (content: string): void => {
		const deduplicated = duplicateSuppressor ? duplicateSuppressor.push(content) : content;
		reportContent(contentFilter ? contentFilter.push(deduplicated) : deduplicated);
	};

	try {
		let request = prepared.request;
		let attempt = 0;

		for (;;) {
			let toolCallsThisAttempt = 0;
			let signal: CompletionSignal | undefined;

			await prepared.client.streamChatCompletion(
				request,
				{
					onContent: handleContent,

					onThinking: (text: string) => {
						reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
						handleThinking(thinkingFilter ? thinkingFilter.push(text) : text, state, progress);
					},

					onToolCall: (toolCall: DeepSeekToolCall) => {
						toolCallsThisAttempt += 1;
						reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
						handleToolCall(toolCall, state, progress, Boolean(contentFilter));
					},

					onError: (error: Error) => {
						throw createUserFacingError(error);
					},

					onDone: (completionSignal?: CompletionSignal) => {
						signal = completionSignal;
					},

					onUsage: (usage) => {
						const charsPerToken = prepared.hasNativeImages
							? getCharsPerToken()
							: updateCharsPerToken(prepared.totalRequestChars, usage, getCharsPerToken());
						if (!prepared.hasNativeImages) {
							setCharsPerToken(charsPerToken);
						}
						prepared.cacheDiagnostics.onUsage(usage, charsPerToken);
						reportCopilotContextUsage(progress, usage, prepared.requestKind);
					},
				},
				token,
			);

			const verificationSaidContinue = await shouldVerifyCompletion(
				prepared,
				request,
				visibleText,
				signal,
				canAutoContinue,
				autoContinue,
				attempt,
				toolCallsThisAttempt,
				token,
			);

			const decision = decideContinuation({
				config: autoContinue,
				signal,
				attempt,
				emittedToolCalls: toolCallsThisAttempt,
				cancelled: token.isCancellationRequested,
				verificationSaidContinue,
			});

			if (!decision.continue || !canAutoContinue) {
				logContinuationDecision(prepared, decision.reason, attempt);
				break;
			}

			attempt += 1;
			logContinuation(prepared, attempt, visibleText.length);
			request = buildContinuationRequest(prepared.request, visibleText, autoContinue.prompt);
			// The partial answer is now replayed as history, so duplicate suppression starts
			// fresh: otherwise the suppressed reasoning would be released on the next attempt.
			duplicateSuppressor?.reset();
		}

		flushFilters(prepared, {
			duplicateSuppressor,
			contentFilter,
			thinkingFilter,
			handleContent,
			handleThinking: (text) => handleThinking(text, state, progress),
			state,
			progress,
		});

		reportInitialResponseNoticeIfPending(progress, state, initialResponseNotice);
		if (token.isCancellationRequested) {
			reportSkippedReplayMarkerIfNeeded(prepared, state, 'cancelled');
		} else {
			reportReplayMarkerOnce(prepared, progress, state, 'done');
		}
		finalizeReplayDiagnostics(prepared.trailingToolResultIds, state, prepared.cacheDiagnostics);
	} catch (error) {
		reportSkippedReplayMarkerIfNeeded(
			prepared,
			state,
			token.isCancellationRequested ? 'cancelled' : 'stream-error',
			error,
		);
		throw error;
	} finally {
		cancelListener.dispose();
	}
}

/** Run the optional "is the task finished?" check before stopping a clean answer. */
async function shouldVerifyCompletion(
	prepared: PreparedChatRequest,
	request: DeepSeekRequest,
	partialText: string,
	signal: CompletionSignal | undefined,
	canAutoContinue: boolean,
	config: AutoContinueConfig,
	attempt: number,
	toolCallsThisAttempt: number,
	token: vscode.CancellationToken,
): Promise<boolean> {
	if (
		!canAutoContinue ||
		!config.verifyCompletion ||
		signal?.truncated ||
		toolCallsThisAttempt > 0 ||
		attempt >= config.maxAttempts ||
		token.isCancellationRequested ||
		!partialText.trim() ||
		// When gated, only spend a request if the answer actually looks cut off.
		(config.verifyOnlyWhenUnfinished && !looksUnfinished(partialText))
	) {
		return false;
	}

	let reply = '';
	try {
		await prepared.client.streamChatCompletion(
			buildCompletionCheckRequest(request, partialText),
			{
				onContent: (text: string) => {
					reply += text;
				},
				onThinking: () => {},
				onToolCall: () => {},
				onError: (error: Error) => {
					throw error;
				},
				onDone: () => {},
			},
			token,
		);
	} catch (error) {
		// Fail open: a broken check must not keep the conversation looping.
		logger.warn(formatRequestLogLine(prepared.requestKind, 'completion check failed'), error);
		return false;
	}

	const done = isCompletionCheckDone(reply);
	logger.debug(
		formatRequestLogLine(
			prepared.requestKind,
			`completion check: ${done ? 'done' : 'continue'} (${reply.trim().slice(0, 40)})`,
		),
	);
	return !done;
}

function flushFilters(
	prepared: PreparedChatRequest,
	context: {
		duplicateSuppressor: DuplicateContentSuppressor | undefined;
		contentFilter: LeakedMarkerFilter | undefined;
		thinkingFilter: LeakedMarkerFilter | undefined;
		handleContent: (content: string) => void;
		handleThinking: (text: string) => void;
		state: ResponseStreamState;
		progress: vscode.Progress<vscode.LanguageModelResponsePart>;
	},
): void {
	const duplicateTail = context.duplicateSuppressor?.flush() ?? '';
	if (duplicateTail) {
		context.handleContent(duplicateTail);
	}
	const contentTail = context.contentFilter?.flush() ?? '';
	if (contentTail) {
		context.handleContent(contentTail);
	}
	const thinkingTail = context.thinkingFilter?.flush() ?? '';
	if (thinkingTail) {
		context.handleThinking(thinkingTail);
	}
	logStrippedMarkers(prepared, context.contentFilter, context.thinkingFilter);
}

function logContinuation(
	prepared: PreparedChatRequest,
	attempt: number,
	visibleChars: number,
): void {
	logger.info(
		formatRequestLogLine(
			prepared.requestKind,
			`output looked truncated after ${visibleChars} chars; continuing (attempt ${attempt})`,
		),
	);
}

function logContinuationDecision(
	prepared: PreparedChatRequest,
	reason: string,
	attempt: number,
): void {
	if (attempt === 0) {
		return;
	}
	logger.info(
		formatRequestLogLine(
			prepared.requestKind,
			`auto-continue stopped: ${reason} after ${attempt} attempt(s)`,
		),
	);
}

function reportInitialResponseNoticeIfPending(
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	initialResponseNotice: string | undefined,
): void {
	reportInitialResponseNoticeOnce(progress, state, initialResponseNotice);
}

function logStrippedMarkers(
	prepared: PreparedChatRequest,
	contentFilter: LeakedMarkerFilter | undefined,
	thinkingFilter: LeakedMarkerFilter | undefined,
): void {
	const contentChars = contentFilter?.strippedChars() ?? 0;
	const thinkingChars = thinkingFilter?.strippedChars() ?? 0;
	if (contentChars === 0 && thinkingChars === 0) {
		return;
	}
	logger.debug(
		formatRequestLogLine(
			prepared.requestKind,
			`stripped leaked markers: content=${contentChars} chars thinking=${thinkingChars} chars`,
		),
	);
}

function reportInitialResponseNoticeOnce(
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	initialResponseNotice: string | undefined,
): void {
	if (!initialResponseNotice || state.initialResponseNoticeReported) {
		return;
	}
	state.initialResponseNoticeReported = true;
	progress.report(new vscode.LanguageModelTextPart(initialResponseNotice));
}

function reportReplayMarkerOnce(
	prepared: PreparedChatRequest,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	trigger: ReplayMarkerReportTrigger,
): void {
	if (state.replayMarkerReported) {
		return;
	}
	state.replayMarkerReported = true;
	reportReplayMarker(prepared, progress, state, trigger);
}

function reportSkippedReplayMarkerIfNeeded(
	prepared: PreparedChatRequest,
	state: ResponseStreamState,
	reason: 'cancelled' | 'stream-error',
	error?: unknown,
): void {
	if (state.replayMarkerReported) {
		return;
	}
	state.replayMarkerReported = true;
	prepared.cacheDiagnostics.onReplayMarkerReport({
		status: 'skipped',
		reason,
		visionTextChars: prepared.visionMarkerTextChars,
		reasoningTextChars: state.accumulatedReasoning.length || undefined,
		error,
	});
}

function reportReplayMarker(
	prepared: PreparedChatRequest,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	state: ResponseStreamState,
	trigger: ReplayMarkerReportTrigger,
): void {
	const metadata = getReplayMarkerMetadata(prepared, state);
	if (!hasReplayMarkerMetadata(metadata)) {
		prepared.cacheDiagnostics.onReplayMarkerReport({
			status: 'skipped',
			trigger,
			reason: 'no-replay-data',
			visionTextChars: prepared.visionMarkerTextChars,
			reasoningTextChars: state.accumulatedReasoning.length || undefined,
		});
		return;
	}

	try {
		const markerPart = createReplayMarkerPart(metadata);
		progress.report(markerPart);
		prepared.cacheDiagnostics.onReplayMarkerReport({
			status: 'reported',
			trigger,
			markerBytes: markerPart.data.byteLength,
			visionTextChars: prepared.visionMarkerTextChars,
			reasoningTextChars: state.accumulatedReasoning.length || undefined,
		});
	} catch (error) {
		prepared.cacheDiagnostics.onReplayMarkerReport({
			status: 'failed',
			trigger,
			visionTextChars: prepared.visionMarkerTextChars,
			reasoningTextChars: state.accumulatedReasoning.length || undefined,
			error,
		});
		logger.warn(
			formatRequestLogLine(prepared.requestKind, 'Failed to report replay marker'),
			error,
		);
	}
}

function getReplayMarkerMetadata(
	prepared: PreparedChatRequest,
	state: ResponseStreamState,
): ReplayMarkerMetadata {
	return {
		...prepared.replayMarkerMetadata,
		reasoningText: state.accumulatedReasoning || undefined,
	};
}

function handleThinking(
	text: string,
	state: ResponseStreamState,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
): void {
	if (!text) {
		return;
	}
	state.accumulatedReasoning += text;

	// LanguageModelThinkingPart is a proposed API; the project root augmentation provides types.
	progress.report(
		new vscode.LanguageModelThinkingPart(text) as unknown as vscode.LanguageModelResponsePart,
	);
}

function handleToolCall(
	toolCall: DeepSeekToolCall,
	state: ResponseStreamState,
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	filterLeakedMarkers: boolean,
): void {
	state.emittedToolCallIds.push(toolCall.id);

	const args = parseToolCallArguments(toolCall.function.arguments, filterLeakedMarkers);
	progress.report(new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.function.name, args));
}

/**
 * Parse tool call arguments.
 *
 * A gateway that appends leaked markup after the JSON body turns a valid call into an
 * unparsable one, which breaks the tool call and can end the turn early. Retry once with
 * the markup removed; when the original text already parses it is used unchanged so real
 * argument content is never rewritten.
 *
 * Markup can also land *inside* a string value (for example a `task_complete` summary,
 * which the host then renders as the reply). Successful parses therefore still have
 * trailing markers removed from their string values.
 */
function parseToolCallArguments(raw: string, filterLeakedMarkers: boolean): object {
	if (!filterLeakedMarkers) {
		return parseJsonOrEmpty(raw);
	}

	try {
		return sanitizeArgumentStrings(JSON.parse(raw)) as object;
	} catch {
		// Fall through and retry with the markup removed.
	}

	const sanitized = stripLeakedMarkers(raw);
	if (sanitized !== raw) {
		try {
			return sanitizeArgumentStrings(JSON.parse(sanitized)) as object;
		} catch {
			// Fall through to the empty-arguments fallback below.
		}
	}

	logger.warn('Failed to parse tool call arguments', raw.slice(0, 200));
	return {};
}

function parseJsonOrEmpty(raw: string): object {
	try {
		return JSON.parse(raw);
	} catch (error) {
		logger.warn('Failed to parse tool call arguments', error);
		return {};
	}
}

/** Remove leaked trailing markers from every string value in the arguments. */
function sanitizeArgumentStrings(value: unknown, depth = 0): unknown {
	if (depth > 8) {
		return value;
	}
	if (typeof value === 'string') {
		return stripTrailingMarkers(value);
	}
	if (Array.isArray(value)) {
		return value.map((item) => sanitizeArgumentStrings(item, depth + 1));
	}
	if (value && typeof value === 'object') {
		const result: Record<string, unknown> = {};
		for (const [key, item] of Object.entries(value)) {
			result[key] = sanitizeArgumentStrings(item, depth + 1);
		}
		return result;
	}
	return value;
}

function finalizeReplayDiagnostics(
	trailingToolResultIds: readonly string[],
	state: ResponseStreamState,
	cacheDiagnostics: CacheDiagnosticsRun,
): void {
	cacheDiagnostics.onDone({
		reasoningTextChars: state.accumulatedReasoning.length,
		emittedToolCalls: state.emittedToolCallIds.length,
		trailingToolResults: trailingToolResultIds.length,
	});
}

function updateCharsPerToken(
	totalRequestChars: number,
	usage: DeepSeekUsage,
	charsPerToken: number,
): number {
	if (totalRequestChars > 0 && usage.prompt_tokens > 0) {
		const observedRatio = totalRequestChars / usage.prompt_tokens;
		return charsPerToken * 0.7 + observedRatio * 0.3;
	}
	return charsPerToken;
}

function reportCopilotContextUsage(
	progress: vscode.Progress<vscode.LanguageModelResponsePart>,
	usage: DeepSeekUsage,
	requestKind: RequestKind,
): void {
	const data = {
		prompt_tokens: usage.prompt_tokens,
		completion_tokens: usage.completion_tokens,
		total_tokens: usage.total_tokens,
		prompt_tokens_details: {
			cached_tokens: usage.prompt_cache_hit_tokens ?? 0,
		},
	};

	try {
		progress.report(
			new vscode.LanguageModelDataPart(
				new TextEncoder().encode(JSON.stringify(data)),
				COPILOT_USAGE_DATA_PART_MIME,
			),
		);
	} catch (error) {
		logger.warn(formatRequestLogLine(requestKind, 'Failed to report usage data'), error);
	}
}
