import { createHash } from 'node:crypto';
import type { VisionImagePart } from './types';

const DEFAULT_MAX_ENTRIES = 128;
const DEFAULT_MAX_CHARS = 4_000_000;

export interface VisionDescriptionCache {
	get(key: string): string | undefined;
	set(key: string, value: string): void;
	clear(): void;
	size(): number;
}

/**
 * Bounded LRU cache for image descriptions.
 *
 * Hosts do not always replay the `toolVision` marker metadata, so the same tool
 * result image can be resolved again on every request of a long running agent
 * session. Keying on the image bytes keeps those repeats off the vision model.
 */
export function createVisionDescriptionCache(
	maxEntries = DEFAULT_MAX_ENTRIES,
	maxChars = DEFAULT_MAX_CHARS,
): VisionDescriptionCache {
	const entries = new Map<string, string>();
	let totalChars = 0;

	const drop = (key: string): void => {
		const value = entries.get(key);
		if (value === undefined) {
			return;
		}
		totalChars -= key.length + value.length;
		entries.delete(key);
	};

	return {
		get(key: string): string | undefined {
			const value = entries.get(key);
			if (value === undefined) {
				return undefined;
			}
			entries.delete(key);
			entries.set(key, value);
			return value;
		},

		set(key: string, value: string): void {
			drop(key);
			entries.set(key, value);
			totalChars += key.length + value.length;
			while (entries.size > maxEntries || totalChars > maxChars) {
				const oldest = entries.keys().next();
				if (oldest.done) {
					break;
				}
				drop(oldest.value);
			}
		},

		clear(): void {
			entries.clear();
			totalChars = 0;
		},

		size(): number {
			return entries.size;
		},
	};
}

export function createVisionImageCacheKey(input: {
	describer: string;
	prompt: string;
	images: readonly VisionImagePart[];
}): string {
	const hash = createHash('sha256');
	hash.update(input.describer);
	hash.update('\0');
	hash.update(input.prompt);
	for (const image of input.images) {
		hash.update('\0');
		hash.update(image.mimeType.toLowerCase());
		hash.update('\0');
		hash.update(image.data);
	}
	return hash.digest('base64url');
}

const descriptionCache = createVisionDescriptionCache();

export function getCachedVisionDescription(key: string): string | undefined {
	return descriptionCache.get(key);
}

export function storeCachedVisionDescription(key: string, description: string): void {
	descriptionCache.set(key, description);
}

export function clearVisionDescriptionCache(): void {
	descriptionCache.clear();
}
