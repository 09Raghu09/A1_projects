// C++ program for implementation of KMP pattern searching
// algorithm
#include <bits/stdc++.h>
#include <string>
#include "bitvector.hpp"

/*
 Construct bwt
 */
std::string bwt_construction(std::string const & text)
{
    std::string bwt;
    std::vector<uint64_t> sa(text.size() + 1);
    
    // Manipulates the values of sa.begin() to sa.end()
    // increases sequentially value, starting with 0
    std::iota(sa.begin(), sa.end(), 0);
    
    // sorting with lambda function for comparing
    std::sort(sa.begin(), sa.end(), [&text](uint64_t a, uint64_t b) -> bool
    {
        while (a < text.size() && b < text.size() && text[a] == text[b])
        {
            ++a, ++b;
        }
        if (b == text.size())
            return false;
        if (a == text.size())
            return true;
        return text[a] < text[b];
    });
    
    for (auto x : sa)
    {
        if (!x)
            bwt += "$";
        else
            bwt += text[x-1];
    }
    return bwt;
}

// convert char into int
size_t to_index(char const chr) {
    switch (chr)
    {
        case '$': return 0;
        case 'A': return 1;
        case 'C': return 2;
        case 'G': return 3;
        case 'T': return 4;
        default:
            throw std::logic_error{"There is an illegal character in your text."};
    }
}


std::vector<uint16_t> compute_count_table(std::string const & bwt) {
    std::vector<uint16_t> count_table(5); // the prefix count table.
    for (auto chr : bwt) {
        // which positions in count_table need to be increased?
        for (size_t i = to_index(chr) + 1; i < 5; ++i)
            ++count_table[i]; // increase position i by 1.
    }
    return count_table;
}


struct occurrence_table
{
    // The list of bitvectors:
    std::vector<Bitvector> data;
    // We want that 5 bitvectors are filled depending on the bwt,
    // so let's customise the constructor of occurrence_table:
    occurrence_table(std::string const & bwt)
    {
        // resize the 5 bitvectors to the length of the bwt:
        data.resize(5, Bitvector((bwt.size() + 63)/ 64));
        // fill the bitvectors
        for (size_t i = 0; i < bwt.size(); ++i)
            data[to_index(bwt[i])].write(i, 1);
        for (Bitvector & bitv : data)
            bitv.construct(3, 6);
    }
    size_t read(char const chr, size_t const i) const
    {
        return data[to_index(chr)].rank(i + 1);
    }
};

size_t count(std::string const & P,
             std::string const & bwt,
             std::vector<uint16_t> const & C,
             occurrence_table const & Occ)
{
    int64_t i = P.size() - 1;
    size_t a = 0;
    size_t b = bwt.size() - 1;
    while ((a <= b) && (i >= 0))
    {
        char c = P[i];
        a = C[to_index(c)] + (a ? Occ.read(c, a - 1) : 0);
        b = C[to_index(c)] + Occ.read(c, b) - 1;
        i = i - 1;
    }
    if (b < a)
        return 0;
    else
        return (b - a + 1);
}

/*
 Open fasta-file, scan and return reads
 */
std::vector<std::string> get_reads(std::string input_file, int reads_number) {
    std::vector<std::string> reads_vector;
    
    // Opening reads-file
    std::ifstream input(input_file);
    if (!input.good()) {
        std::cerr << "Error opening: " << input_file << " . You have failed." << std::endl;
    }
    
    // Iterating over reads
    std::string line, id, DNA_sequence;
    int counter = 0;
    while (std::getline(input, line)) {
        // Stop if given reads_number is satisfied
        if (counter == reads_number) {
            break;
        }
        
        // line may be empty -> *must* ignore blank lines
        if(line.empty())
            continue;
        
        // ID - line
        if (line[0] == '>') {
            // output previous line before overwriting id
            // but ONLY if id actually contains something
            if(!id.empty()) {
                reads_vector.push_back(DNA_sequence);
                counter += 1;
            }
            id = line.substr(1);
            DNA_sequence.clear();
        }
        // if not ID
        else {
            DNA_sequence += line;
        }
    }
    // output final entry
    // but ONLY if id actually contains something
    if(!id.empty() && counter < reads_number) {
        reads_vector.push_back(DNA_sequence);
    }
    return reads_vector;
}


// Driver program to test above function
int main(int argc, char *argv[]) {
    // get reference text
    std::string reference_file = "./../read_data/reference.fasta";
    std::string reference_text = get_reads(reference_file, 1).front();
    // compute the bwt, C and Occ:
    std::string bwt = bwt_construction(reference_text);
    std::vector<uint16_t> C = compute_count_table(bwt);
    occurrence_table Occ(bwt);
    
    
    // get reads
    int reads_number = std::stoi(argv[1]);
    std::string reads_file = "./../read_data/reads.fasta";
    
    std::vector<std::string> reads_vector = get_reads(reads_file, reads_number);
    std::cout << "Choosen reads number: " << reads_vector.size() << std::endl;
    
    // iterate over reads and compute their number of matches
    for (std::vector<std::string>::iterator it = reads_vector.begin(); it != reads_vector.end(); ++it) {
        // get count for current read
        //std::cout << count(*it, bwt, C, Occ) << '\n'; // prints 2
        
        // no printout needed -> but still compute!
        count(*it, bwt, C, Occ);
    }
    
    
    /*
    std::string text{"mississippi"};
    // compute the bwt, C and Occ:
    std::string bwt = bwt_construction(text);
    std::vector<uint16_t> C = compute_count_table(bwt);
    occurrence_table Occ(bwt);
    
    
    std::cout << count("ssi", bwt, C, Occ) << '\n'; // prints 2
    std::cout << count("ppi", bwt, C, Occ) << '\n'; // prints 1
     */
}
