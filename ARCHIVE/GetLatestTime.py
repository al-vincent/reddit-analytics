def main():
    infile = 'part-r-00000'
    earliest_time = 1502647916
    latest_time = 0
    with open(infile, 'r') as f:
        for line in f:
            items = line.split("\t")
            #print(len(items))
            if len(items) == 3:
                timestamp = int(items[2].rstrip('\n'))
                if timestamp > latest_time:
                    latest_time = timestamp
                if timestamp < earliest_time:
                    earliest_time = timestamp
            else:
                continue
    
    print("Earliest time: " + str(earliest_time))
    print("Latest time: " + str(latest_time))

if __name__ == '__main__':
    main()
