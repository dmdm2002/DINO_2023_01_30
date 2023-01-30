import csv
import sys
import traceback
import time


def error_logger(image_name: str, place: str, function: str, e: Exception):
    """
    :param image_name: Error Image Name
    :param place: Error .py file Name
    :param function: Error function name
    :param e: Error Type
    :var stop: can process stop
    :return: sys.exit (exit program), You can change the code to continue if you want.
    """
    stop = True
    # assert stop is bool, 'Only boolean type is available for stop in error_logger class'

    error_log = [image_name, place, function, e]
    f = open('./Error_Log.csv', "a")
    writer = csv.writer(f, lineterminator='\n')

    writer.writerow(error_log)
    f.close()

    print(traceback.format_exc())
    if stop:
        time.sleep(0.5)
        sys.exit(f'ERROR!! [ IMAGE: {image_name}, PLACE: {place}, FUNCTION: {function}, TYPE: {e}')
    else:
        pass