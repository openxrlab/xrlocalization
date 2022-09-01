import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
