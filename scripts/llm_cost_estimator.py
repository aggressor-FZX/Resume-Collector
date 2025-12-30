#!/usr/bin/env python3
"""Estimate LLM API costs given token usage and pricing parameters.

Usage examples:
  python scripts/llm_cost_estimator.py --calls 10000 --avg-input-tokens 131100 --avg-output-tokens 16400 --input-price 0.02 --output-price 0.02

Notes:
- Prices are interpreted as USD per 1k tokens by default (use --price-per-token to input per-token prices).
- The script prints a short table including Total Context, Max Output, Input Price, Output Price, Cache Read/Write and audio fields if provided.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass

@dataclass
class CostParams:
    calls: int
    avg_input_tokens: float
    avg_output_tokens: float
    input_price_per_1k: float
    output_price_per_1k: float
    cache_read_price: float | None = None
    cache_write_price: float | None = None
    input_audio_price: float | None = None
    input_audio_cache_price: float | None = None

    def input_cost(self) -> float:
        return (self.calls * self.avg_input_tokens / 1000.0) * self.input_price_per_1k

    def output_cost(self) -> float:
        return (self.calls * self.avg_output_tokens / 1000.0) * self.output_price_per_1k

    def total_cost(self) -> float:
        c = self.input_cost() + self.output_cost()
        # cache and audio optional additions
        if self.cache_read_price is not None:
            c += self.calls * self.cache_read_price
        if self.cache_write_price is not None:
            c += self.calls * self.cache_write_price
        if self.input_audio_price is not None:
            c += self.calls * self.input_audio_price
        if self.input_audio_cache_price is not None:
            c += self.calls * self.input_audio_cache_price
        return c


def format_usd(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--calls', type=int, default=1, help='Number of model calls (examples)')
    p.add_argument('--avg-input-tokens', type=float, default=0.0, help='Average prompt tokens per call')
    p.add_argument('--avg-output-tokens', type=float, default=0.0, help='Average output tokens per call')
    p.add_argument('--input-price', type=float, default=0.02, help='Input price (USD per 1k tokens)')
    p.add_argument('--output-price', type=float, default=0.02, help='Output price (USD per 1k tokens)')
    p.add_argument('--cache-read-price', type=float, default=None, help='Cache read price per call (USD)')
    p.add_argument('--cache-write-price', type=float, default=None, help='Cache write price per call (USD)')
    p.add_argument('--input-audio-price', type=float, default=None, help='Input audio price per unit (USD)')
    p.add_argument('--input-audio-cache-price', type=float, default=None, help='Input audio cache price per call (USD)')
    p.add_argument('--price-per-token', action='store_true', help='Interpret --input-price/--output-price as per-token (not per 1k)')
    p.add_argument('--show-per-call', action='store_true', help='Show per-call breakdown')
    args = p.parse_args()

    params = CostParams(
        calls=args.calls,
        avg_input_tokens=args.avg_input_tokens,
        avg_output_tokens=args.avg_output_tokens,
        input_price_per_1k=(args.input_price * 1000.0) if args.price_per_token else args.input_price,
        output_price_per_1k=(args.output_price * 1000.0) if args.price_per_token else args.output_price,
        cache_read_price=args.cache_read_price,
        cache_write_price=args.cache_write_price,
        input_audio_price=args.input_audio_price,
        input_audio_cache_price=args.input_audio_cache_price,
    )

    input_cost = params.input_cost()
    output_cost = params.output_cost()
    total_cost = params.total_cost()

    print('\nLLM Cost Estimation Summary')
    print('---------------------------')
    print(f'Total calls: {params.calls:,}')
    print(f'Total context (avg input tokens per call): {params.avg_input_tokens:,.1f}')
    print(f'Max output (avg output tokens per call): {params.avg_output_tokens:,.1f}')
    print(f'Input Price (USD per 1k tokens): {params.input_price_per_1k:.6f}')
    print(f'Output Price (USD per 1k tokens): {params.output_price_per_1k:.6f}')
    print('\nBreakdown:')
    print(f' - Input cost: {format_usd(input_cost)}')
    print(f' - Output cost: {format_usd(output_cost)}')
    if params.cache_read_price is not None:
        print(f' - Cache read total: {format_usd(params.cache_read_price * params.calls)} ({format_usd(params.cache_read_price)} per call)')
    if params.cache_write_price is not None:
        print(f' - Cache write total: {format_usd(params.cache_write_price * params.calls)} ({format_usd(params.cache_write_price)} per call)')
    if params.input_audio_price is not None:
        print(f' - Input audio total: {format_usd(params.input_audio_price * params.calls)}')
    if params.input_audio_cache_price is not None:
        print(f' - Input audio cache total: {format_usd(params.input_audio_cache_price * params.calls)}')
    print('\nTotal estimated cost: ', format_usd(total_cost))

    if args.show_per_call and params.calls > 0:
        per_call = total_cost / params.calls
        print(f'Per-call estimated cost: {format_usd(per_call)}')

    print('\nNotes:')
    print(' - Prices default to USD per 1k tokens unless --price-per-token is set.')
    print(' - Use actual token counts for best estimates. Tokenization approx: ~4 chars = 1 token as a rough guide.')


if __name__ == '__main__':
    main()
