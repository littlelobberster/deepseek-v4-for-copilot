import vscode from 'vscode';
import { getShowRunningIndicatorEnabled } from '../config';

const REFRESH_INTERVAL_MS = 1000;

interface RunningIndicatorHandle {
	end(): void;
}

let item: vscode.StatusBarItem | undefined;
let activeCount = 0;
let startedAt = 0;
let timer: ReturnType<typeof setInterval> | undefined;
let disposed = false;

/**
 * Mark the start of a chat request.
 *
 * The provider API has no way to decorate the chat input box, so the only place an
 * extension can show live progress is the status bar. Calls are reference counted
 * because several requests can overlap.
 */
export function beginRunningIndicator(): RunningIndicatorHandle {
	if (!getShowRunningIndicatorEnabled()) {
		return { end: () => {} };
	}

	activeCount += 1;
	if (activeCount === 1) {
		startedAt = Date.now();
		update();
		timer = setInterval(update, REFRESH_INTERVAL_MS);
		// Never keep the extension host alive just for the spinner.
		if (typeof timer === 'object' && typeof timer.unref === 'function') {
			timer.unref();
		}
		getItem()?.show();
	}

	let ended = false;
	return {
		end: () => {
			if (ended) {
				return;
			}
			ended = true;
			endRunningIndicator();
		},
	};
}

export function disposeRunningIndicator(): void {
	disposed = true;
	stopTimer();
	item?.dispose();
	item = undefined;
	activeCount = 0;
}

function endRunningIndicator(): void {
	activeCount = Math.max(0, activeCount - 1);
	if (activeCount > 0) {
		return;
	}
	stopTimer();
	item?.hide();
}

function stopTimer(): void {
	if (timer !== undefined) {
		clearInterval(timer);
		timer = undefined;
	}
}

function update(): void {
	const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
	const running = getItem();
	if (running) {
		running.text = `$(sync~spin) DeepSeek · ${formatElapsed(seconds)}`;
		running.tooltip = `DeepSeek is working (${formatElapsed(seconds)})`;
	}
}

function formatElapsed(seconds: number): string {
	if (seconds < 60) {
		return `${seconds}s`;
	}
	const minutes = Math.floor(seconds / 60);
	const remainder = seconds % 60;
	return `${minutes}m ${remainder.toString().padStart(2, '0')}s`;
}

function getItem(): vscode.StatusBarItem | undefined {
	if (disposed) {
		return undefined;
	}
	if (!item) {
		item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 1000);
		// The Remote indicator blue, so the item reads as "busy" at a glance.
		item.backgroundColor = new vscode.ThemeColor('statusBarItem.remoteBackground');
		item.color = new vscode.ThemeColor('statusBarItem.remoteForeground');
		item.name = 'DeepSeek';
	}
	return item;
}
