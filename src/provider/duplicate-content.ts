/**
 * Some gateways emit the same text twice: once on the reasoning channel and again as the
 * reply body, with no tags to mark it. The body then reads like thinking and the turn
 * looks like it ended without an answer.
 *
 * This classifies a content stream as duplicate while it still matches the reasoning
 * already received, and releases it as soon as it diverges. A real answer diverges
 * immediately, so normal replies are unaffected.
 */

/** How much duplicate text to accept before deciding without waiting for the end. */
const MAX_HELD_DUPLICATE_CHARS = 64 * 1024;
/** How long to wait for a reasoning channel before treating the body as a real reply. */
const REASONING_WAIT_CHARS = 2048;

export type ContentClassification = 'pending' | 'duplicate' | 'reply';

export class DuplicateContentSuppressor {
	private held = '';
	private state: ContentClassification = 'pending';
	private lastReasoningLength = -1;

	constructor(
		private readonly getReasoning: () => string,
		private readonly hasReasoningChannel: boolean,
	) {}

	get classification(): ContentClassification {
		return this.state;
	}

	/**
	 * Feed a content chunk.
	 *
	 * Returns the text that should be shown, which is empty while the chunk still matches
	 * the reasoning prefix.
	 */
	push(chunk: string): string {
		if (!chunk) {
			return '';
		}
		if (this.state === 'reply') {
			return chunk;
		}
		if (!this.hasReasoningChannel) {
			// The model has no reasoning channel, so nothing can be duplicated.
			return this.releaseAll(chunk);
		}

		const reasoning = this.getReasoning();
		const reasoningGrew = reasoning.length !== this.lastReasoningLength;
		this.lastReasoningLength = reasoning.length;

		this.held += chunk;

		if (!reasoning) {
			// Reasoning has not started yet; wait briefly so an interleaved duplicate is caught.
			if (this.held.length >= REASONING_WAIT_CHARS) {
				return this.releaseAll();
			}
			return '';
		}

		if (this.held.length <= reasoning.length) {
			if (isPrefix(this.held, reasoning)) {
				if (this.held.length > MAX_HELD_DUPLICATE_CHARS) {
					// Still identical after a lot of text: it is the reasoning, not an answer.
					this.state = 'duplicate';
					this.held = '';
				}
				return '';
			}
			return this.releaseAll();
		}

		// The held text is longer than the reasoning received so far.
		if (!this.held.startsWith(reasoning)) {
			return this.releaseAll();
		}

		// It repeats the reasoning and continues. Only treat the remainder as the answer once
		// the reasoning has stopped growing, otherwise the reasoning may simply be behind.
		if (reasoningGrew) {
			return '';
		}
		const answer = this.held.slice(reasoning.length);
		this.state = 'reply';
		this.held = '';
		return answer;
	}

	/** Flush at end of stream; duplicate text is discarded, an appended answer is kept. */
	flush(): string {
		if (this.state === 'reply') {
			return '';
		}

		const reasoning = this.getReasoning();
		const held = this.held;
		this.held = '';

		if (!held) {
			this.state = 'duplicate';
			return '';
		}

		if (!reasoning) {
			// Reasoning never arrived: this is ordinary reply text.
			this.state = 'reply';
			return held;
		}

		// The body repeated the reasoning and then continued: the remainder is the answer.
		if (held.startsWith(reasoning)) {
			const remainder = held.slice(reasoning.length);
			if (remainder) {
				this.state = 'reply';
				return remainder;
			}
			this.state = 'duplicate';
			return '';
		}

		// The body is still a prefix of the reasoning: pure duplication.
		if (reasoning.startsWith(held)) {
			this.state = 'duplicate';
			return '';
		}

		this.state = 'reply';
		return held;
	}

	private releaseAll(extra = ''): string {
		this.state = 'reply';
		const released = this.held + extra;
		this.held = '';
		return released;
	}
}

function isPrefix(candidate: string, reasoning: string): boolean {
	const length = Math.min(candidate.length, reasoning.length);
	if (length === 0) {
		return true;
	}
	return candidate.slice(0, length) === reasoning.slice(0, length);
}
