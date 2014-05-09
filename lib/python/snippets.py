# Todo:
-   First create vocabulary for test set, then use only those tokens from the training set that are also in the test set


# In chunks
with open(...) as f:
    while True:
        chunk = list(islice(f, n))
        if not chunk:
            break
        process_data(chunk)

# Lazy chunks
def yield_chunks(file_object, chunk_size=1024):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data        

f = open('really_big_file.dat')
for chunk in yield_chunks(f):
    process_data(chunk)        

# line-based files are already lazy generated:
for line in open('really_big_file.dat'):
    process_data(line)        
