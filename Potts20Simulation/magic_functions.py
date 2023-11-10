
def magic_function(f):
    return f+10

def process_frame(f):
    # changed your logic here as I couldn't repro it
    return f, magic_function(f)
