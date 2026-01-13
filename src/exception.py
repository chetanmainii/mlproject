import sys
from src.logger import logging
import logging

def error_message(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    
    if exc_tb is None:
        return str(error)
    
    file_name=exc_tb.tb_frame.f_code.co_filename
    message=f"exception caused in python script {file_name} at line number {exc_tb.tb_lineno} \
    error message {str(error)}"
    return message
    
class CustomException(Exception):
    def __init__(self,message,error_detail:sys):
        super().__init__(message) 
        self.message=error_message(message,error_detail)   
        
    def __str__(self):
        return self.message

    