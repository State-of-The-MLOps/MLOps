import logging
import logging.handlers

L = logging.getLogger('snowdeer_log')
L.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d -  %(funcName)20s()]\n\t%(message)s",
    datefmt='%y-%m-%d %H:%M:%S'
)

fileHandler = logging.FileHandler('./log.txt')
streamHandler = logging.StreamHandler()

fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

L.addHandler(fileHandler)
L.addHandler(streamHandler)
