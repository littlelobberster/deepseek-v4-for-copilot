/**
 * Some OpenAI-compatible gateways put reasoning into the `content` field wrapped in
 * `<thinking>...</thinking>` tags instead of using `reasoning_content`. Those tags then
 * land in the assistant text, are shown in the chat and get replayed as context on the
 * next request, so they are stripped from both the live stream and stored history.
 */

const OPEN_TAG = '<thinking';
const CLOSE_TAG = '</thinking>';
const CLOSE_TOKEN = '</thinking>';

const OPEN_TAG_PATTERN = /<thinking(?=[\s/>]|$)/i;
const COMPLETE_BLOCK_PATTERN = /<thinking\b[^>]*>[\s\S]*?<\/thinking\s*>/gi;
const STRAY_CLOSE_TAG_PATTERN = /<\/thinking\s*>/gi;

export interface ThinkingTagFilter {
	/** Feed a streamed content chunk; returns the text that is safe to show now. */
	push(chunk: string): string;
	/** Drain the pending buffer at the end of the response. */
	flush(): string;
	/** Total characters removed by the filter so far. */
	strippedChars(): number;
}

/** Remove leaked `<thinking>` blocks and stray closing tags from a complete text. */
export function stripThinkingTags(text: string): string {
	if (!text.includes('<')) {
		return text;
	}
	const lowered = text.toLowerCase();
	let result = text;
	if (lowered.includes(OPEN_TAG)) {
		result = result.replace(COMPLETE_BLOCK_PATTERN, '');
	}
	if (lowered.includes('</thinking')) {
		result = result.replace(STRAY_CLOSE_TAG_PATTERN, '');
	}
	return result;
}

/**
 * Streaming-safe variant of {@link stripThinkingTags}.
 *
 * Tags arrive split across chunks, so text that could still be the start of a tag is
 * held back until the rest of the chunk arrives.
 */
export function createThinkingTagFilter(): ThinkingTagFilter {
	let pending = '';
	let insideBlock = false;
	let stripped = 0;

	const drain = (final: boolean): string => {
		let out = '';

		for (;;) {
			if (insideBlock) {
				const closeIndex = findTagStart(pending, CLOSE_TAG);
				if (closeIndex < 0) {
					if (final) {
						stripped += pending.length;
						pending = '';
						insideBlock = false;
						return out;
					}
					const keep = partialPrefixLength(pending, CLOSE_TOKEN);
					stripped += pending.length - keep;
					pending = keep > 0 ? pending.slice(pending.length - keep) : '';
					return out;
				}

				const tagEnd = pending.indexOf('>', closeIndex);
				if (tagEnd < 0) {
					if (final) {
						stripped += pending.length;
						pending = '';
						insideBlock = false;
					} else {
						stripped += closeIndex;
						pending = pending.slice(closeIndex);
					}
					return out;
				}

				stripped += tagEnd + 1;
				pending = pending.slice(tagEnd + 1);
				insideBlock = false;
				continue;
			}

			const openTag = findOpenTag(pending);
			let emitEnd = pending.length;
			if (openTag && openTag.index < emitEnd) {
				emitEnd = openTag.index;
			}
			const openHold = pending.length - partialPrefixLength(pending, OPEN_TAG);
			if (openHold < emitEnd) {
				emitEnd = openHold;
			}
			const closeHold = pending.length - partialPrefixLength(pending, CLOSE_TOKEN);
			if (closeHold < emitEnd) {
				emitEnd = closeHold;
			}

			if (emitEnd > 0) {
				out += pending.slice(0, emitEnd).replace(STRAY_CLOSE_TAG_PATTERN, '');
				pending = pending.slice(emitEnd);
				continue;
			}

			if (!openTag || openTag.index !== 0) {
				return out;
			}

			if (openTag.end < 0) {
				if (final) {
					stripped += pending.length;
					pending = '';
				}
				return out;
			}

			stripped += openTag.end;
			pending = pending.slice(openTag.end);
			insideBlock = true;
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
			insideBlock = false;
			return out;
		},

		strippedChars(): number {
			return stripped;
		},
	};
}

function findOpenTag(text: string): { index: number; end: number } | undefined {
	const match = OPEN_TAG_PATTERN.exec(text);
	if (!match) {
		return undefined;
	}
	const end = text.indexOf('>', match.index);
	return { index: match.index, end: end < 0 ? -1 : end + 1 };
}

function findTagStart(text: string, tag: string): number {
	return text.toLowerCase().indexOf(tag);
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
