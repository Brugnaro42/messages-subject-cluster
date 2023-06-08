import logging

def get_logger(logger, filename: str = 'experiment.log', level: int = 10):
    """
    Get logger object

    Parameters
    ----------
    logger
        logger object
    filename: str
        log file name. Default: 'experiment.log'
    level: int
        log level. Default: 10 - DEBUG
    Returns
    -------
    object
        configured logger object
    """

    logger.setLevel(level)

    string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(string)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler(filename=filename, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger