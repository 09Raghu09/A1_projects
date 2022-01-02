#include <iostream>
#include <iostream>
#include <string>
#include <fstream>


#include "bloom_filter.hpp"

/*
 Open fasta-file, scan and return reads
 */
std::vector<std::string> get_fasta(std::string input_file, int reads_number) {
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

    input.close();
    return reads_vector;
}

std::vector<std::string> get_kmers(std::string input_file, int ksize) {
    std::string reference_text = get_fasta(input_file, 1).front();

    std::vector<std::string> kmer_vector;
    int text_length = reference_text.size();

    for (int i=0; i<text_length - ksize; ++i) {
        // add substring of position i, and length ksize
        kmer_vector.push_back(reference_text.substr(i, ksize));
    }
    return kmer_vector;
}

// Driver program to test above function
int main(int argc, char *argv[]) {
    std::string mouse_x_chr_fasta = "./../read_data/mm_ref_GRCm38.p6_chrX.fa";
    std::string human_x_chr_fasta = "./../read_data/hs_ref_GRCh37.p9_chrX.fa";

    // Set parameters
    int kmer_size = std::stoi(argv[1]);
    double false_positive_rate = std::atof(argv[2]);
    std::cout << false_positive_rate << std::endl;

    bloom_parameters parameters;
    // Maximum tolerable false positive probability?
    parameters.false_positive_probability = false_positive_rate;

    // Take mouse x-chromosome as reference-text
    std::vector<std::string> mouse_kmer_vector = get_kmers(mouse_x_chr_fasta, kmer_size);
    std::cout << "Number of kmers for mouse: " << mouse_kmer_vector.size() << std::endl;

    // How many elements roughly do we expect to insert?
    //parameters.projected_element_count = mouse_kmer_vector.size();
    parameters.projected_element_count = mouse_kmer_vector.size();

    // Simple randomizer (optional)
    //parameters.random_seed = 0xA5A5A5A5;

    if (!parameters)
    {
        std::cout << "Error - Invalid set of bloom filter parameters!" << std::endl;
        return 1;
    }


    const clock_t begin_time = clock();
    parameters.compute_optimal_parameters();
    //Instantiate Bloom Filter
    bloom_filter filter(parameters);
    //compressible_bloom_filter filter(parameters);
    // Insert kmers into Bloom Filter
    filter.insert(mouse_kmer_vector.begin(), mouse_kmer_vector.end());
    std::cout << "Building Filter Time: "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    /*
    for (std::vector<std::string>::iterator it = mouse_kmer_vector.begin(); it != mouse_kmer_vector.end(); ++it) {
        filter.insert(*it);
    }
     */


    // Get human kmers
    std::vector<std::string> human_kmer_vector = get_kmers(human_x_chr_fasta, kmer_size);
    std::cout << "Number of kmers for human: " << human_kmer_vector.size() << std::endl;

    // Query the existence of kmers
    int counter = 0;
    for (std::vector<std::string>::iterator it = human_kmer_vector.begin(); it != human_kmer_vector.end(); ++it) {
        if (filter.contains(*it)) {
            //std::cout << *it << "\n" << std::endl;
            counter += 1;
        }
    }
    std::cout << "Number of positive hits: " << counter << std::endl;
    std::cout << "Desired fpr: " << false_positive_rate << " - achieved fpr: " << filter.effective_fpp() << std::endl;
    std::cout << "Size of filter: " << filter.size() << std::endl;


    return 0;
}
