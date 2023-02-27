"""
The entry of text to speechg.
"""
import argparse
from speech.text2speech.orchestrator import TTSOrchestrator


def parse_argument():
    parser = argparse.ArgumentParser(description="Speech to text module.")
    sub_parsers = parser.add_subparsers(help='mode', dest='mode', required=True)

    # Text 2 speech inference parameters.
    inference_parser = sub_parsers.add_parser(name='inference', help='Text to speech inference.')
    inference_parser.add_argument('-t', '--text', required=True, help='The text content.')
    inference_parser.add_argument('-sr', '--sample-rate', default=16000, help='The sample rate of the input audio.')
    inference_parser.add_argument('-l', '--language', choices=['ru', 'en', 'es', 'de', 'fr'], default='en',
        help='The language of voice.')
    inference_parser.add_argument('-r', '--read-mode', choices=['incremental', 'one_shot'], default='one_shot',
        help='read mode: incremental is fast to response, but there are pause between chunked sentences; ' +
            'one_shot is more natural, but delay is high for long response.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    module = None
    if args.mode == 'inference':
        module = TTSOrchestrator(
            text=args.text, sample_rate=args.sample_rate, language=args.language, read_mode=args.read_mode)
    if module:
        module.process()
