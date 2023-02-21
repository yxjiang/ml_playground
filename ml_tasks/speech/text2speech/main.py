"""
The entry of text to speechg.
"""
import argparse
import orchestrator

def parse_argument():
    parser = argparse.ArgumentParser(description="Speech to text module.")
    sub_parsers = parser.add_subparsers(help='mode', dest='mode', required=True)

    # Text 2 speech inference parameters.
    inference_parser = sub_parsers.add_parser(name='inference', help='Text to speech inference.')
    inference_parser.add_argument('-t', '--text', required=True, help='The text content.')
    inference_parser.add_argument('-sr', '--sample-rate', default=16000, help='The sample rate of the input audio.')
    inference_parser.add_argument('-l', '--language', choices=['ru', 'en', 'es'], default='en', help='The language of voice.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    module = None
    if args.mode == 'inference':
        module = orchestrator.TTSOrchestrator(text=args.text, sample_rate=args.sample_rate, language=args.language)
    if module:
        module.run()
