import type { CompletionSignal, DeepSeekRequest } from '../types';

/** Default nudge appended after a truncated answer. */
export const DEFAULT_CONTINUATION_PROMPT =
	'Your previous reply was cut off. Continue from exactly where it stopped. Do not repeat any text you already wrote, and do not add a preamble.';

/** Prompt used by the optional completion check. */
export const COMPLETION_CHECK_PROMPT =
	'Consider the work so far. If every part of the task is finished, reply with exactly "DONE". If anything is still missing, reply with exactly "CONTINUE". Reply with one word only.';

export interface AutoContinueConfig {
	enabled: boolean;
	maxAttempts: number;
	prompt: string;
	verifyCompletion: boolean;
}

export interface ContinuationDecision {
	continue: boolean;
	reason: 'disabled' | 'cancelled' | 'complete' | 'tool-calls' | 'max-attempts' | 'truncated';
}

/** Whether a truncated (or unfinished) response should be continued automatically. */
export function decideContinuation(input: {
	config: AutoContinueConfig;
	signal: CompletionSignal | undefined;
	attempt: number;
	emittedToolCalls: number;
	cancelled: boolean;
	verificationSaidContinue: boolean;
}): ContinuationDecision {
	if (!input.config.enabled) {
		return { continue: false, reason: 'disabled' };
	}
	if (input.cancelled) {
		return { continue: false, reason: 'cancelled' };
	}
	// A tool call ends this response on purpose: the host runs the tool and comes back.
	if (input.emittedToolCalls > 0) {
		return { continue: false, reason: 'tool-calls' };
	}
	if (input.attempt >= input.config.maxAttempts) {
		return { continue: false, reason: 'max-attempts' };
	}
	if (input.signal?.truncated) {
		return { continue: true, reason: 'truncated' };
	}
	if (input.verificationSaidContinue) {
		return { continue: true, reason: 'truncated' };
	}
	return { continue: false, reason: 'complete' };
}

/**
 * Build the follow-up request for a truncated answer.
 *
 * The partial answer is replayed as the assistant turn and a short nudge asks the model to
 * resume. Keeping the partial text is what lets the continuation pick up mid-sentence
 * instead of restarting the answer.
 */
export function buildContinuationRequest(
	request: DeepSeekRequest,
	partialText: string,
	prompt: string,
): DeepSeekRequest {
	const messages = [...request.messages];
	const partial = partialText.replace(/\s+$/, '');
	if (partial) {
		messages.push({ role: 'assistant', content: partial });
	}
	messages.push({ role: 'user', content: prompt });
	return { ...request, messages };
}

/** Build the optional "is the task finished?" check. */
export function buildCompletionCheckRequest(
	request: DeepSeekRequest,
	partialText: string,
): DeepSeekRequest {
	const messages = [...request.messages];
	const partial = partialText.replace(/\s+$/, '');
	if (partial) {
		messages.push({ role: 'assistant', content: partial });
	}
	messages.push({ role: 'user', content: COMPLETION_CHECK_PROMPT });
	return {
		...request,
		messages,
		// The check must not stream into the reply.
		stream: false,
		tools: undefined,
		tool_choice: undefined,
	};
}

/** Interpret the completion check reply; anything that is not a clear DONE keeps working. */
export function isCompletionCheckDone(reply: string): boolean {
	const normalized = reply.trim().toUpperCase();
	if (!normalized) {
		return false;
	}
	if (normalized.startsWith('DONE')) {
		return true;
	}
	// Unknown output: treat as done so a chatty model cannot loop forever.
	return !normalized.includes('CONTINUE');
}
