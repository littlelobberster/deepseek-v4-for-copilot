/**
 * Gateways in front of DeepSeek models sometimes leak their own control markup into the
 * text stream instead of using a dedicated reasoning channel. Observed leaks include
 * `<thinking>…</thinking>`, `<thought>…</thought>`, `<summary>`, `<now>`, `<content>` and
 * raw special tokens such as `</｜｜DSML｜｜ parameter>`.
 *
 * Removing markup is only safe with narrow rules, because the same strings are real
 * content elsewhere: C# writes `/// <summary>…</summary>`, HTML uses
 * `<details><summary>…</summary></details>`, and users quote markup when asking about it.
 * So this filter:
 *
 * - never touches fenced code blocks or inline code spans,
 * - drops the *content* of reasoning tags (reasoning already goes to the thinking channel),
 * - drops other scaffolding tags but keeps their text, and
 * - leaves unknown tags alone so real HTML/XML keeps working.
 */

/** Tags whose whole block is leaked reasoning, not answer text. */
const REASONING_TAGS = new Set([
	'thinking',
	'thought',
	'think',
	'analytical',
	'analytic',
	'reasoning',
	'reasoning_scratchpad',
	'visual_thinking',
]);

/** Tags that wrap answer text, so only the tag itself is removed. */
const SCAFFOLDING_TAGS = new Set([
	'summary',
	'now',
	'content',
	'status',
	'invoke',
	'review_begin',
	'clarification',
	'query',
	'analysis',
	'scratchpad',
	'plan',
]);

const SPECIAL_TOKEN_PATTERN = /^<\/?[|｜][^<>\n]{1,60}>/; // Tolerate spacing variants such as `</ summary>` that gateways emit.
const TAG_PATTERN = /^<\s*(\/?)\s*([A-Za-z][\w.-]{0,40})(\s[^<>\n]{0,80})?>/;
const SPECIAL_TOKEN_START_PATTERN = /^<\/?[|｜][^<>\n]{0,60}$/;
const FENCE_LINE_PATTERN = /^ {0,3}(`{3,}|~{3,})/;
const FENCE_PREFIX_PATTERN = /^ {0,3}[`~]{0,2}$/;
// Only `//` style comments are protected. Markdown headings (`#`) and bullets (`*`) are
// ordinary output lines, so leaked tags on them must still be removed.
const COMMENT_LINE_PATTERN = /^\s*\/\/+/;
const TOOL_SCAFFOLD_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*-(?:mcp-server|server)(?:-[a-z0-9_-]+)+$/;

/** How much of the current line is kept to recognise comment-style lines. */
const LINE_PREFIX_CHARS = 16;
/** Longest tag we may hold back while waiting for the rest of the chunk. */
const MAX_TAG_CHARS = 128;
/** Unclosed backticks longer than this are treated as literal text instead of waiting. */
const MAX_PENDING_CODE_SPAN_CHARS = 4096;
/**
 * Upper bound on how much text a single reasoning block may swallow. Gateways sometimes
 * prefix reasoning with a bare `<thinking>` and never close it; without a bound the rest
 * of the message would disappear.
 */
const MAX_DROPPED_BLOCK_CHARS = 4000;

export interface LeakedMarkerFilterOptions {
	/**
	 * Drop the *content* of reasoning blocks (`<thinking>…</thinking>`).
	 *
	 * Keep this off for the thinking channel and for replayed reasoning: that text already
	 * is reasoning, so only the bare tags should be removed. It stays on for reply content,
	 * where a leaked reasoning block is not part of the answer.
	 */
	dropReasoningBlocks?: boolean;
	/**
	 * Where reasoning that leaked into reply content should go instead of the body.
	 *
	 * Gateways sometimes prefix reasoning with a bare `<thinking>` and never close it. That
	 * text is reasoning, not answer, so it is handed here (the thinking channel) rather than
	 * rendered as body text. Without a handler such blocks are dropped.
	 */
	onLeakedReasoning?: (text: string) => void;
}

export interface LeakedMarkerFilter {
	/** Feed a streamed chunk; returns the text that is safe to show now. */
	push(chunk: string): string;
	/** Drain the pending buffer at the end of the response. */
	flush(): string;
	/** Total characters removed by the filter so far. */
	strippedChars(): number;
}

/** Remove leaked gateway markup from a complete text. */
export function stripLeakedMarkers(text: string, options: LeakedMarkerFilterOptions = {}): string {
	const filter = createLeakedMarkerFilter(options);
	return filter.push(text) + filter.flush();
}

const TRAILING_MARKER_PATTERN =
	/(?:<\s*\/?\s*(?:thinking|thought|think|analytical|analytic|reasoning|reasoning_scratchpad|visual_thinking|summary|now|content|status|invoke|review_begin|clarification|query|analysis|scratchpad|plan)\b[^<>\n]{0,80}>|<\s*\/?\s*[|｜][^<>\n]{1,60}>)$/i;

/**
 * Strip marker tags that leaked onto the **end** of a value.
 *
 * Gateways append their closing marker after the whole completion, which lands inside
 * tool call arguments such as a `task_complete` summary. Such arguments still parse as
 * valid JSON, so nothing else filters them, yet the host renders that summary as the
 * assistant reply and the tag reaches the user. Only trailing markers are removed:
 * mid-value text, and values ending inside an unterminated code fence, stay untouched.
 */
export function stripTrailingMarkers(text: string): string {
	if (!text.includes('<') || hasUnterminatedFence(text) || endsInsideDetails(text)) {
		return text;
	}

	const endsWithNewline = text.endsWith('\n');
	let body = text.replace(/\s+$/, '');
	let changed = false;

	for (;;) {
		// A comment line owns its markup (for example `/// <summary>…</summary>`), so a
		// trailing tag there is real content, not a leak.
		if (isCommentLine(lastLineOf(body))) {
			break;
		}
		const next = body.replace(TRAILING_MARKER_PATTERN, '').replace(/\s+$/, '');
		if (next === body) {
			break;
		}
		body = next;
		changed = true;
	}

	if (!changed) {
		return text;
	}
	return endsWithNewline && body ? `${body}\n` : body;
}

function lastLineOf(text: string): string {
	const index = text.lastIndexOf('\n');
	return index < 0 ? text : text.slice(index + 1);
}

function isCommentLine(line: string): boolean {
	return /^\s*(?:\/\/|<!--)/.test(line);
}

/** A value that genuinely ends inside an open `<details>` owns its `<summary>`. */
function endsInsideDetails(text: string): boolean {
	const lowered = text.toLowerCase();
	const lastOpen = lowered.lastIndexOf('<details');
	if (lastOpen < 0) {
		return false;
	}
	const lastClose = lowered.lastIndexOf('</details');
	if (lastClose > lastOpen) {
		return false;
	}
	// Only guard when the open block really reaches the end of the value; an inline
	// example mentioned earlier in the text must not disable cleanup.
	return text.length - lastOpen < 300;
}

function hasUnterminatedFence(text: string): boolean {
	let count = 0;
	for (let index = text.indexOf('```'); index >= 0; index = text.indexOf('```', index + 3)) {
		count += 1;
	}
	return count % 2 === 1;
}

interface TagMatch {
	raw: string;
	name: string;
	closing: boolean;
	attrs: boolean;
}

/**
 * Streaming-safe marker filter.
 *
 * Tags arrive split across chunks (`<thin` + `king>`), so an incomplete tag is held back
 * until the rest of the chunk arrives.
 */
export function createLeakedMarkerFilter(
	options: LeakedMarkerFilterOptions = {},
): LeakedMarkerFilter {
	const dropReasoningBlocks = options.dropReasoningBlocks !== false;
	const onLeakedReasoning = options.onLeakedReasoning;

	let pending = '';
	let blockTag: string | undefined;
	let blockBuffer = '';
	let fence: { char: string; len: number } | undefined;
	let atLineStart = true;
	let linePrefix = '';
	let lineEmittedAny = false;
	let suppressNewline = false;
	let openDetails = 0;
	let stripped = 0;

	const drain = (final: boolean): string => {
		const out: string[] = [];

		const emit = (text: string): void => {
			if (!text) {
				return;
			}
			out.push(text);
			lineEmittedAny = true;
			suppressNewline = false;
			linePrefix = (linePrefix + text).slice(0, LINE_PREFIX_CHARS);
		};

		const drop = (length: number): void => {
			stripped += length;
			if (!lineEmittedAny) {
				// The dropped markup was the only thing on this line.
				suppressNewline = true;
			}
		};

		/**
		 * Reasoning that leaked into the body: hand it to the thinking channel, never to the
		 * body. Without a handler it is simply dropped.
		 */
		const divertLeakedReasoning = (text: string): void => {
			if (!text) {
				return;
			}
			stripped += text.length;
			onLeakedReasoning?.(text);
		};

		const endLine = (): void => {
			if (suppressNewline) {
				suppressNewline = false;
			} else {
				out.push('\n');
			}
			atLineStart = true;
			linePrefix = '';
			lineEmittedAny = false;
		};

		for (;;) {
			if (blockTag) {
				const closeIndex = indexOfIgnoreCase(pending, `</${blockTag}`);
				if (closeIndex >= 0) {
					const closeEnd = pending.indexOf('>', closeIndex);
					if (closeEnd >= 0) {
						// A complete block: drop it, including the opening tag we buffered.
						stripped += blockBuffer.length + closeEnd + 1;
						blockBuffer = '';
						pending = pending.slice(closeEnd + 1);
						blockTag = undefined;
						suppressNewline = !lineEmittedAny;
						continue;
					}
					if (!final) {
						// Wait for the rest of the closing tag.
						blockBuffer += pending.slice(0, closeIndex);
						pending = pending.slice(closeIndex);
						return out.join('');
					}
				}

				// No closing tag yet: buffer the text so it can be restored when the block
				// turns out to be unterminated (gateways sometimes emit a bare `<thinking>`).
				const keep = partialPrefixLength(pending, `</${blockTag}>`);
				blockBuffer += pending.slice(0, pending.length - keep);
				pending = pending.slice(pending.length - keep);

				if (blockBuffer.length > MAX_DROPPED_BLOCK_CHARS) {
					// Do not eat the rest of the message. If this is body text, hand the
					// reasoning to the thinking channel; otherwise drop it as before. The block
					// stays open so the remaining text is handled the same way.
					divertLeakedReasoning(stripLeakedMarkers(blockBuffer, { dropReasoningBlocks: false }));
					blockBuffer = '';
					return out.join('');
				}

				if (!final) {
					// A closing tag may still arrive, so keep buffering instead of guessing where
					// the reasoning ends. Guessing here would make the streamed result differ from
					// a single-shot parse of the same text.
					return out.join('');
				}

				// End of stream without a closing tag: the buffered text is reasoning. The answer
				// usually starts after the first blank line, so split there and keep parsing the
				// remainder as reply text instead of dropping it.
				const restore = blockBuffer + pending;
				blockBuffer = '';
				pending = '';
				blockTag = undefined;

				const paragraphBreak = restore.indexOf('\n\n');
				if (paragraphBreak >= 0) {
					divertLeakedReasoning(
						stripLeakedMarkers(restore.slice(0, paragraphBreak), {
							dropReasoningBlocks: false,
						}),
					);
					pending = restore.slice(paragraphBreak + 2);
					continue;
				}

				divertLeakedReasoning(stripLeakedMarkers(restore, { dropReasoningBlocks: false }));
				return out.join('');
			}

			if (pending.length === 0) {
				return out.join('');
			}

			if (fence) {
				const newline = pending.indexOf('\n');
				if (atLineStart) {
					const closing = detectClosingFence(pending, fence, final);
					if (closing === 'unknown') {
						return out.join('');
					}
					if (closing === 'yes') {
						const stop = newline < 0 ? pending.length : newline + 1;
						out.push(pending.slice(0, stop));
						pending = pending.slice(stop);
						fence = undefined;
						atLineStart = newline < 0;
						linePrefix = '';
						lineEmittedAny = false;
						suppressNewline = false;
						continue;
					}
					atLineStart = false;
				}
				if (newline < 0) {
					out.push(pending);
					pending = '';
					return out.join('');
				}
				out.push(pending.slice(0, newline + 1));
				pending = pending.slice(newline + 1);
				atLineStart = true;
				linePrefix = '';
				lineEmittedAny = false;
				suppressNewline = false;
				continue;
			}

			if (atLineStart) {
				const opening = FENCE_LINE_PATTERN.exec(pending);
				if (opening) {
					const newline = pending.indexOf('\n');
					if (newline < 0 && !final) {
						return out.join('');
					}
					fence = { char: opening[1][0], len: opening[1].length };
					const stop = newline < 0 ? pending.length : newline + 1;
					out.push(pending.slice(0, stop));
					pending = pending.slice(stop);
					atLineStart = newline < 0;
					linePrefix = '';
					lineEmittedAny = false;
					suppressNewline = false;
					continue;
				}
				if (!final && FENCE_PREFIX_PATTERN.test(pending)) {
					return out.join('');
				}
				atLineStart = false;
			}

			const specialIndex = findNextSpecialIndex(pending);
			if (specialIndex > 0) {
				emit(pending.slice(0, specialIndex));
				pending = pending.slice(specialIndex);
				continue;
			}

			const char = pending[0];

			if (char === '\n') {
				pending = pending.slice(1);
				endLine();
				continue;
			}

			if (char === '`') {
				const run = countLeadingRun(pending, '`');
				if (run === pending.length && !final) {
					return out.join('');
				}
				const newline = pending.indexOf('\n');
				const limit = newline < 0 ? pending.length : newline;
				const closeIndex = indexOfBacktickRun(pending, run, limit, run);
				if (closeIndex >= 0) {
					emit(pending.slice(0, closeIndex + run));
					pending = pending.slice(closeIndex + run);
					continue;
				}
				if (newline < 0 && !final && pending.length < MAX_PENDING_CODE_SPAN_CHARS) {
					return out.join('');
				}
				emit(pending.slice(0, run));
				pending = pending.slice(run);
				continue;
			}

			if (char === '<') {
				const specialToken = SPECIAL_TOKEN_PATTERN.exec(pending);
				if (specialToken) {
					drop(specialToken[0].length);
					pending = pending.slice(specialToken[0].length);
					continue;
				}

				// HTML comments are real content: emit them verbatim and do not filter inside.
				if (pending.startsWith('<!--')) {
					const commentEnd = pending.indexOf('-->');
					if (commentEnd < 0) {
						if (!final && pending.length < MAX_PENDING_CODE_SPAN_CHARS) {
							return out.join('');
						}
						emit('<');
						pending = pending.slice(1);
						continue;
					}
					emit(pending.slice(0, commentEnd + 3));
					pending = pending.slice(commentEnd + 3);
					continue;
				}

				const tag = matchTag(pending);
				if (!tag) {
					if (!final && pending.length < MAX_TAG_CHARS && couldBecomeMarkup(pending)) {
						return out.join('');
					}
					emit('<');
					pending = pending.slice(1);
					continue;
				}

				if (isReasoningTag(tag.name)) {
					if (tag.closing) {
						drop(tag.raw.length);
						pending = pending.slice(tag.raw.length);
						continue;
					}
					if (hasAttributes(tag)) {
						// Looks like prose ("use <think about it>"), not a marker.
						emit(tag.raw);
						pending = pending.slice(tag.raw.length);
						continue;
					}
					// Only a bare tag at the start of a line is treated as leaked reasoning
					// whose body should be dropped. Everywhere else the tag goes away and the
					// text stays, which keeps reasoning content intact.
					const structural = dropReasoningBlocks && !lineEmittedAny;
					if (!structural) {
						drop(tag.raw.length);
						pending = pending.slice(tag.raw.length);
						continue;
					}

					const closeIndex = indexOfIgnoreCase(pending, `</${tag.name}`, tag.raw.length);
					if (closeIndex >= 0) {
						const closeEnd = pending.indexOf('>', closeIndex);
						if (closeEnd >= 0) {
							drop(closeEnd + 1);
							pending = pending.slice(closeEnd + 1);
							continue;
						}
						if (!final) {
							blockBuffer = pending;
							pending = '';
							blockTag = tag.name;
							continue;
						}
						emit(pending);
						pending = '';
						return out.join('');
					}

					if (!final) {
						// Drop the tag now so an unterminated block restores only real text.
						drop(tag.raw.length);
						pending = pending.slice(tag.raw.length);
						blockBuffer = '';
						blockTag = tag.name;
						continue;
					}

					drop(tag.raw.length);
					pending = pending.slice(tag.raw.length);
					continue;
				}

				if (tag.name === 'details') {
					emit(tag.raw);
					openDetails = Math.max(0, openDetails + (tag.closing ? -1 : 1));
					pending = pending.slice(tag.raw.length);
					continue;
				}

				if (isScaffoldingTag(tag.name)) {
					// Real content keeps its markup: C# doc comments and HTML summary elements.
					const inComment = COMMENT_LINE_PATTERN.test(linePrefix);
					const htmlSummary = tag.name === 'summary' && openDetails > 0;
					if (inComment || htmlSummary) {
						emit(tag.raw);
					} else {
						drop(tag.raw.length);
					}
					pending = pending.slice(tag.raw.length);
					continue;
				}

				emit(tag.raw);
				pending = pending.slice(tag.raw.length);
				continue;
			}

			emit(char);
			pending = pending.slice(1);
		}
	};

	return {
		push(chunk: string): string {
			if (!chunk) {
				return '';
			}
			pending += chunk;
			return drain(false);
		},

		flush(): string {
			const out = drain(true);
			// drain(true) restores an unterminated reasoning block, so nothing is buffered here.
			blockBuffer = '';
			pending = '';
			blockTag = undefined;
			fence = undefined;
			return out;
		},

		strippedChars(): number {
			return stripped;
		},
	};
}

function matchTag(text: string): TagMatch | undefined {
	const match = TAG_PATTERN.exec(text);
	if (!match) {
		return undefined;
	}
	return {
		raw: match[0],
		name: match[2].toLowerCase(),
		closing: match[1] === '/',
		attrs: Boolean(match[3]),
	};
}

/** Whether the partial text could still grow into a tag or special token. */
function couldBecomeMarkup(text: string): boolean {
	if (!text.startsWith('<')) {
		return false;
	}
	if (SPECIAL_TOKEN_START_PATTERN.test(text)) {
		return true;
	}
	if (text.includes('>')) {
		// A complete tag would already have been matched; this is prose.
		return false;
	}
	// Hold any in-progress tag (`<details>` must stay recognisable across chunks) and any
	// in-progress HTML comment, bounded by MAX_TAG_CHARS at the call site.
	return (
		/^<\s*\/?\s*(?:[A-Za-z][\w.-]{0,40})?(?:\s[^<>|\n]{0,80})?$/.test(text) ||
		/^<!?-{0,2}$/.test(text)
	);
}

function isReasoningTag(name: string): boolean {
	return REASONING_TAGS.has(name);
}

function hasAttributes(tag: TagMatch): boolean {
	return tag.attrs;
}

function isScaffoldingTag(name: string): boolean {
	return SCAFFOLDING_TAGS.has(name) || TOOL_SCAFFOLD_PATTERN.test(name);
}

function detectClosingFence(
	text: string,
	fence: { char: string; len: number },
	final: boolean,
): 'yes' | 'no' | 'unknown' {
	let index = 0;
	while (index < 3 && text[index] === ' ') {
		index += 1;
	}

	let run = 0;
	while (text[index + run] === fence.char) {
		run += 1;
		if (index + run >= text.length) {
			return final && run >= fence.len ? 'yes' : 'unknown';
		}
	}

	if (run < fence.len) {
		return 'no';
	}

	const next = text[index + run];
	return next === '\n' || next === '\r' || next === ' ' || next === '\t' ? 'yes' : 'no';
}

function findNextSpecialIndex(text: string): number {
	for (let index = 0; index < text.length; index += 1) {
		const char = text[index];
		if (char === '<' || char === '`' || char === '\n') {
			return index;
		}
	}
	return -1;
}

function indexOfIgnoreCase(text: string, needle: string, fromIndex = 0): number {
	return text.toLowerCase().indexOf(needle, fromIndex);
}

/** Length of the longest suffix of `text` that is still an incomplete prefix of `tag`. */
function partialPrefixLength(text: string, tag: string): number {
	const max = Math.min(text.length, tag.length - 1);
	for (let length = max; length > 0; length -= 1) {
		if (text.slice(text.length - length).toLowerCase() === tag.slice(0, length).toLowerCase()) {
			return length;
		}
	}
	return 0;
}

function countLeadingRun(text: string, char: string): number {
	let count = 0;
	while (count < text.length && text[count] === char) {
		count += 1;
	}
	return count;
}

/** Index of a run of at least `length` backticks within `[start, limit)`. */
function indexOfBacktickRun(text: string, start: number, limit: number, length: number): number {
	for (let index = start; index < limit; index += 1) {
		if (text[index] !== '`') {
			continue;
		}
		let run = 0;
		while (index + run < limit && text[index + run] === '`') {
			run += 1;
		}
		if (run >= length) {
			return index;
		}
		index += run - 1;
	}
	return -1;
}
