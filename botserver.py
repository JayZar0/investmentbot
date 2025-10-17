

## this file will be used use the functions of the bot and tell the user which
## stocks to invest in. I may also make everything automated if I get too lazy
## to do the investments myself.

## I may also turn this to an api that can send graphs over to the front end.

# This function will be called after the stock data has been created. It will
# choose the stock based on the numbers that are shown in the stock data creation.
def select_stock():
    return NotImplementedError

# This function will be called on the start of the stock market opening to create
# the prediction of the stock market and which direction the stocks will take.
def create_stock_data():
    return NotImplementedError

# This function will be used to correct any errors that were made during the
# predictions that were made during the stock opening.
# It will only be called if the error margin was over 5%.
def correct_errors():
    return NotImplementedError

# This function will be used to save all the data of the stocks to create a
# data validation to see if the model is headed in the right direction.
def end_of_trades():
    return NotImplementedError

def generate_prediction_graph():
    return NotImplementedError