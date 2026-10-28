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

const SPECIAL_TOKEN_PATTERN = /^<\/?[|｜][^<>\n]{1,60}>/;
const TAG_PATTERN = /^<(\/?)([A-Za-z][\w.-]{0,40})(?:\s[^<>\n]{0,80})?>/;
const TAG_START_PATTERN = /^<\/?(?:[A-Za-z][\w.-]{0,40})?$/;
const SPECIAL_TOKEN_START_PATTERN = /^<\/?[|｜][^<>\n]{0,60}$/;
const FENCE_LINE_PATTERN = /^ {0,3}(`{3,}|~{3,})/;
const FENCE_PREFIX_PATTERN = /^ {0,3}[`~]{0,2}$/;
const COMMENT_LINE_PATTERN = /^\s*(?:\/\/|\*|<!--|#)/;
const TOOL_SCAFFOLD_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*-(?:mcp-server|server)(?:-[a-z0-9_-]+)+$/;

/** How much of the current line is kept to recognise comment-style lines. */
const LINE_PREFIX_CHARS = 16;
/** Longest tag we may hold back while waiting for the rest of the chunk. */
const MAX_TAG_CHARS = 128;
/** Unclosed backticks longer than this are treated as literal text instead of waiting. */
const MAX_PENDING_CODE_SPAN_CHARS = 4096;

export interface LeakedMarkerFilter {
	/** Feed a streamed chunk; returns the text that is safe to show now. */
	push(chunk: string): string;
	/** Drain the pending buffer at the end of the response. */
	flush(): string;
	/** Total characters removed by the filter so far. */
	strippedChars(): number;
}

/** Remove leaked gateway markup from a complete text. */
export function stripLeakedMarkers(text: string): string {
	const filter = createLeakedMarkerFilter();
	return filter.push(text) + filter.flush();
}

interface TagMatch {
	raw: string;
	name: string;
	closing: boolean;
}

/**
 * Streaming-safe marker filter.
 *
 * Tags arrive split across chunks (`<thin` + `king>`), so an incomplete tag is held back
 * until the rest of the chunk arrives.
 */
export function createLeakedMarkerFilter(): LeakedMarkerFilter {
	let pending = '';
	let blockTag: string | undefined;
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
				if (closeIndex < 0) {
					const keep = partialPrefixLength(pending, `</${blockTag}>`);
					stripped += pending.length - keep;
					pending = pending.slice(pending.length - keep);
					if (final) {
						stripped += pending.length;
						pending = '';
						blockTag = undefined;
					}
					return out.join('');
				}

				const closeEnd = pending.indexOf('>', closeIndex);
				if (closeEnd < 0) {
					if (!final) {
						stripped += closeIndex;
						pending = pending.slice(closeIndex);
						return out.join('');
					}
					stripped += pending.length;
					pending = '';
					blockTag = undefined;
					continue;
				}

				stripped += closeEnd + 1;
				pending = pending.slice(closeEnd + 1);
				blockTag = undefined;
				suppressNewline = !lineEmittedAny;
				continue;
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
					// Only a bare tag at the start of a line is treated as leaked reasoning.
					// Anything else may be prose about markup, so it is left alone.
					const structural = !lineEmittedAny && !hasAttributes(tag);
					const closeIndex = indexOfIgnoreCase(pending, `</${tag.name}`, tag.raw.length);
					if (closeIndex < 0) {
						if (!final) {
							if (structural) {
								drop(tag.raw.length);
								pending = pending.slice(tag.raw.length);
								blockTag = tag.name;
								continue;
							}
							return out.join('');
						}
						if (structural) {
							drop(tag.raw.length);
							pending = pending.slice(tag.raw.length);
							blockTag = tag.name;
							continue;
						}
						emit(pending);
						pending = '';
						return out.join('');
					}
					const closeEnd = pending.indexOf('>', closeIndex);
					if (closeEnd < 0) {
						if (!final) {
							return out.join('');
						}
						emit(pending);
						pending = '';
						return out.join('');
					}
					drop(closeEnd + 1);
					pending = pending.slice(closeEnd + 1);
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
	};
}

/** Whether the partial text could still grow into a tag or special token. */
function couldBecomeMarkup(text: string): boolean {
	return TAG_START_PATTERN.test(text) || SPECIAL_TOKEN_START_PATTERN.test(text);
}

function isReasoningTag(name: string): boolean {
	return REASONING_TAGS.has(name);
}

function hasAttributes(tag: TagMatch): boolean {
	return tag.raw.length > tag.name.length + 2;
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
