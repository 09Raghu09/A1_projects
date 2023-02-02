import pysam

bamfile = pysam.AlignmentFile(".bam", "rb") #TODO: add path to file here
outfile = pysam.AlignmentFile("Users/isabel/Desktop/tumor.bam", 'rb')
i = 0
for read in bamfile.fetch():
    if i <= 1000:
        outfile.write(read)
    else:
        break
outfile.close()
bamfile.close()