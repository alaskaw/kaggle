import csv, sys

if len(sys.argv) <> 3:
    print >>sys.stderr, 'Wrong number of arguments. This tool will print first n records from a comma separated CSV file.' 
    print >>sys.stderr, 'Usage:' 
    print >>sys.stderr, '       python', sys.argv[0], '<file> <number-of-lines>'
    sys.exit(1)

fileName = sys.argv[1]
n = int(sys.argv[2])

i = 0
out = csv.writer(sys.stdout, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
with open(fileName, 'rb') as csvfile:    
    next(csvfile, None)  # skip header
    for row in csv.reader(csvfile, delimiter=',', quotechar='"'):
        i += 1
        if i > n: break
        else:
            out.writerow(row)

