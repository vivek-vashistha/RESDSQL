#!/usr/bin/env python3
"""
Script to explore the Spider training dataset questions used to train the T5 base model.

The T5 base model was trained on the Spider dataset training set.
This script helps you explore and extract questions from that dataset.
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load_training_data(data_path="data/spider/train_spider.json"):
    """Load the Spider training dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def show_statistics(data):
    """Display statistics about the training dataset."""
    print("=" * 60)
    print("SPIDER TRAINING DATASET STATISTICS")
    print("=" * 60)
    print(f"Total training examples: {len(data)}")
    
    # Count by database
    db_counts = defaultdict(int)
    for example in data:
        db_counts[example['db_id']] += 1
    
    print(f"\nNumber of unique databases: {len(db_counts)}")
    print(f"\nTop 10 databases by number of questions:")
    sorted_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (db_id, count) in enumerate(sorted_dbs[:10], 1):
        print(f"  {i}. {db_id}: {count} questions")
    
    print("\n" + "=" * 60)


def show_random_samples(data, num_samples=10):
    """Display random sample questions."""
    print(f"\n{'=' * 60}")
    print(f"RANDOM SAMPLE QUESTIONS ({num_samples} examples)")
    print("=" * 60)
    
    samples = random.sample(data, min(num_samples, len(data)))
    for i, example in enumerate(samples, 1):
        print(f"\n{i}. Question: {example['question']}")
        print(f"   Database: {example['db_id']}")
        print(f"   SQL: {example['query']}")


def show_by_database(data, db_id, limit=10):
    """Show questions for a specific database."""
    matching = [ex for ex in data if ex['db_id'] == db_id]
    
    if not matching:
        print(f"No questions found for database: {db_id}")
        return
    
    print(f"\n{'=' * 60}")
    print(f"QUESTIONS FOR DATABASE: {db_id} ({len(matching)} total)")
    print("=" * 60)
    
    for i, example in enumerate(matching[:limit], 1):
        print(f"\n{i}. {example['question']}")
        print(f"   SQL: {example['query']}")
    
    if len(matching) > limit:
        print(f"\n... and {len(matching) - limit} more questions")


def list_databases(data):
    """List all available databases."""
    db_counts = defaultdict(int)
    for example in data:
        db_counts[example['db_id']] += 1
    
    print("\n" + "=" * 60)
    print("AVAILABLE DATABASES")
    print("=" * 60)
    sorted_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (db_id, count) in enumerate(sorted_dbs, 1):
        print(f"  {i:3d}. {db_id:40s} ({count:4d} questions)")


def export_questions(data, output_file, format='txt'):
    """Export questions to a file."""
    if format == 'txt':
        with open(output_file, 'w') as f:
            f.write("SPIDER TRAINING DATASET QUESTIONS\n")
            f.write("=" * 60 + "\n\n")
            for i, example in enumerate(data, 1):
                f.write(f"{i}. Question: {example['question']}\n")
                f.write(f"   Database: {example['db_id']}\n")
                f.write(f"   SQL: {example['query']}\n\n")
    elif format == 'json':
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} questions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore Spider training dataset questions used to train T5 base model"
    )
    parser.add_argument(
        '--data-path',
        default='data/spider/train_spider.json',
        help='Path to train_spider.json file'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show dataset statistics'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=0,
        help='Show N random sample questions'
    )
    parser.add_argument(
        '--db',
        type=str,
        help='Show questions for a specific database ID'
    )
    parser.add_argument(
        '--list-dbs',
        action='store_true',
        help='List all available databases'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export all questions to a file (specify filename)'
    )
    parser.add_argument(
        '--format',
        choices=['txt', 'json'],
        default='txt',
        help='Export format (txt or json)'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        print(f"Please ensure the Spider training data is available at: {data_path}")
        return
    
    data = load_training_data(str(data_path))
    
    # Execute requested actions
    if args.stats:
        show_statistics(data)
    
    if args.list_dbs:
        list_databases(data)
    
    if args.db:
        show_by_database(data, args.db)
    
    if args.samples > 0:
        show_random_samples(data, args.samples)
    
    if args.export:
        export_questions(data, args.export, args.format)
    
    # If no specific action, show stats and samples
    if not any([args.stats, args.list_dbs, args.db, args.samples, args.export]):
        show_statistics(data)
        show_random_samples(data, 5)
        print("\n" + "=" * 60)
        print("TIP: Use --help to see all available options")
        print("=" * 60)


if __name__ == "__main__":
    main()
