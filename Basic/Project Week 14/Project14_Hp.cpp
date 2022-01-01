/*
 * C++ Program to Implement Suffix Array
 */
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
using namespace std;

class SuffixArray {
private:
    string *text;
    int length;
    int *index;
    string *suffix;

public:
    SuffixArray(string text) {
        this->text = new string[text.length()];
        for (int i = 0; i < text.length(); i++)
        {
            this->text[i] = text.substr(i, 1);
        }
        this->length = text.length();
        this->index = new int[length];
        for (int i = 0; i < length; i++)
        {
            index[i] = i;
        }
        suffix = new string[length];
    }

    void createSuffixArray()
    {
        for(int index = 0; index < length; index++)
        {
            string text = "";
            for (int text_index = index; text_index < length; text_index++)
            {
                text += this->text[text_index];
            }
            suffix[index] = text;
        }
        int back;
        for (int iteration = 1; iteration < length; iteration++)
        {
            string key = suffix[iteration];
            int keyindex = index[iteration];
            for (back = iteration - 1; back >= 0; back--)
            {
                if (suffix[back].compare(key) > 0)
                {
                    suffix[back + 1] = suffix[back];
                    index[back + 1] = index[back];
                }
                else
                {
                    break;
                }
            }
            suffix[back + 1] = key;
            index[back + 1 ] = keyindex;
        }

        cout<<"SUFFIX \t INDEX"<<endl;

        for (int iterate = 0; iterate < length; iterate++)
        {
            cout<<suffix[iterate] << "\t" << index[iterate]<<endl;
        }

    }


    string* get_suffix_pointer() {
        return this->suffix;
    }
};

int main()
{
    string text,text_input;
    cout<<"Enter the Text String: ";
    cin>>text_input;
    text = text_input + text_input;
    int n = text.length();
    SuffixArray suffixarray(text);
    suffixarray.createSuffixArray();

    string* suffix_pointer = suffixarray.get_suffix_pointer();
    for (int iterate = 0; iterate < text.length(); ++iterate) {
        if(suffix_pointer[iterate].length()>=(n/2))
        {
            cout << "The minimum lexicographic rotation of the sentence is " << suffix_pointer[iterate].substr(0,(n/2));
            break;
        }
    }
}