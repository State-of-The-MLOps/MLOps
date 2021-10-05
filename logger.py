import logging
import logging.handlers
from colorlog import ColoredFormatter


L = logging.getLogger("snowdeer_log")
L.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    fmt="%(log_color)s [%(levelname)s] %(reset)s %(asctime)s [%(filename)s:%(lineno)d -  %(funcName)20s()]\n\t%(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)

fileHandler = logging.FileHandler("./log.txt")
streamHandler = logging.StreamHandler()

fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

L.addHandler(fileHandler)
L.addHandler(streamHandler)
