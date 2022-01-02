# Bloom filter based search algorithm

A bloom filter is a space efficient, probabilistic
data structure that is used to tell whether a
member is in a set.

This project was centered on comparing two similar genome sequences, assuming they have many overlapping k-mers.
Two genomic datasets of reasonable size that can be compared (e.g. two full smaller genomes
or only one chromosome of each). You can find genome data here (https://www.ncbi.nlm.nih.gov/guide/howto/dwn-genome/)
Naively, we could find k-mers (substrings of length k) common to s1 and s2 by
comparing every k-mer from one sequence to every k-mer from the other. Such an
approach would take worst-case time Θ( |s1| * |s2| ), which is unacceptable
because interesting DNA sequences range from thousands to billions of characters in
length.

Call s1 the corpus string and s2 the pattern string, and assume we are searching for common
substrings of length k. We first construct a set S of every k-mer in the pattern string. Then, for each kmer in the corpus string, we check whether it occurs in the set S; if so, we have found a match
between pattern and corpus.
