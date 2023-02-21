"""
The entry of speech to text.
"""
import argparse
import orchestrator


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech to text module.")
    sub_parsers = parser.add_subparsers(help='mode', dest='mode', required=True)
    # Training parameters.
    train_model_parser = sub_parsers.add_parser('train', help='train mode: Train the speech to text model.')
    train_model_parser.add_argument('-d', '--training_data_path', required=True, help='The path of the training data folder.')

    # Batch speech recognition parameters.
    batch_mode_parser = sub_parsers.add_parser('batch', help='batch mode: Speech to text for the given audio file.')
    batch_mode_parser.add_argument('-f', '--file_path', required=True, help='Path of the audio file.')
    batch_mode_parser.add_argument('-sr', '--sample_rate', default=8000, help='The sample rate of the output audio.')
    batch_mode_parser.add_argument('-fb', '--frames_per_buffer', default=1024 * 1024 * 4,
        help='Determines how many frames to read at a time in the input stream. It is a trade-off between throughput and latency.')
    batch_mode_parser.add_argument('-d', '--decoder', choices=['greedy', 'libri'], default='libri',
        help='The decoder used to rescore the candidates.')

    # Real time speech recognition parameters.
    realtime_mode_parser = sub_parsers.add_parser('realtime', help='')
    realtime_mode_parser.add_argument('-sr', '--sample_rate', default=8000, help='The sample rate of the output audio.')
    realtime_mode_parser.add_argument('-fb', '--frames_per_buffer', default=2048,
        help='Determines how many frames to read at a time in the input stream. It is a trade-off between throughput and latency.')
    realtime_mode_parser.add_argument('-b', '--buffer_size', default=20,
        help='The number of available frames before sending to the speech to text model.')
    realtime_mode_parser.add_argument('-f', '--file_path', default='tmp_file.wav', help='Temp file to store the stream.')
    realtime_mode_parser.add_argument('-d', '--decoder', choices=['greedy', 'libri'], default='libri',
        help='The decoder used to rescore the candidates.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    module = None
    if args.mode == 'batch':
        module = orchestrator.BatchOrchestrator(args=args)
    elif args.mode == 'realtime':
        module = orchestrator.StreamOrchestrator(args=args)
    if module:
        module.run()