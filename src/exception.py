import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Extracts the exact file name, line number, and error message 
    from Python's system execution information.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    """
    OOP Custom Exception class inheriting from Python's base Exception class.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Formats the error with exact line numbers and file names
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message