import logging


def loggers(name='log'):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  

    
    logfile = name + '.txt'
    fh = logging.FileHandler(logfile, mode='a')  
    fh.setLevel(logging.DEBUG)  

    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  

    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
