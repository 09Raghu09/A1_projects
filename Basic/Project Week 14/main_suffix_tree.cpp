// A C++ program to implement Ukkonen's Suffix Tree Construction
// Here we build generalized suffix tree for given string S
// and it's reverse R, then we find
// longest palindromic substring of given string S

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_set>
#define MAX_CHAR 256
using namespace std;

struct SuffixTreeNode {
	struct SuffixTreeNode *children[MAX_CHAR];

	//pointer to other node via suffix link
	struct SuffixTreeNode *suffixLink;

	/*(start, end) interval specifies the edge, by which the
	node is connected to its parent node. Each edge will
	connect two nodes, one parent and one child, and
	(start, end) interval of a given edge will be stored
	in the child node. */
	int start;
	int *end;

	/*for leaf nodes, it stores the index of suffix for
	the path from root to leaf*/
	int suffixIndex;

	//To store indices of children suffixes in given string
	unordered_set<int> *forwardIndices;

	//To store indices of children suffixes in reversed string
	unordered_set<int> *reverseIndices;
};

typedef struct SuffixTreeNode Node;

char text[20000]; //Input string
Node *root = NULL; //Pointer to root node

/*lastNewNode will point to newly created internal node,
waiting for it's suffix link to be set, which might get
a new suffix link (other than root) in next extension of
same phase. lastNewNode will be set to NULL when last
newly created internal node (if there is any) got it's
suffix link reset to new internal node created in next
extension of same phase. */
Node *lastNewNode = NULL;
Node *activeNode = NULL;

int activeEdge = -1;
int activeLength = 0;

// remainingSuffixCount tells how many suffixes yet to
// be added in tree
int remainingSuffixCount = 0;
int leafEnd = -1;
int *rootEnd = NULL;
int *splitEnd = NULL;
int size = -1; //Length of input string
int size1 = 0; //Size of 1st string
int reverseIndex; //Index of a suffix in reversed string
unordered_set<int>::iterator forwardIndex;

Node *newNode(int start, int *end)
{
	Node *node =(Node*) malloc(sizeof(Node));
	int i;
	for (i = 0; i < MAX_CHAR; i++)
		node->children[i] = NULL;

	/*For root node, suffixLink will be set to NULL
	For internal nodes, suffixLink will be set to root
	by default in current extension and may change in
	next extension*/

	node->suffixLink = root;
	node->start = start;
	node->end = end;

	/*suffixIndex will be set to -1 by default and
	actual suffix index will be set later for leaves
	at the end of all phases*/
	node->suffixIndex = -1;
	node->forwardIndices = new unordered_set<int>;
	node->reverseIndices = new unordered_set<int>;
	return node;
}

int edgeLength(Node *n) {
	if(n == root)
		return 0;
	return *(n->end) - (n->start) + 1;
}

int walkDown(Node *currNode)
{
	/*activePoint change for walk down. If activeLength is greater
	than current edge length, set next internal node as
	activeNode and adjust activeEdge and activeLength
	accordingly to represent same activePoint*/
	if (activeLength >= edgeLength(currNode))
	{
		activeEdge += edgeLength(currNode);
		activeLength -= edgeLength(currNode);
		activeNode = currNode;
		return 1;
	}
	return 0;
}

void extendSuffixTree(int pos)
{
	leafEnd = pos;

	/*Increment remainingSuffixCount indicating that a
	new suffix added to the list of suffixes yet to be
	added in tree*/
	remainingSuffixCount++;

	/*set lastNewNode to NULL while starting a new phase,
	indicating there is no internal node waiting for
	it's suffix link reset in current phase*/
	lastNewNode = NULL;

	//Add all suffixes (yet to be added) one by one in tree
	while(remainingSuffixCount > 0) {

		if (activeLength == 0)
			activeEdge = pos; //APCFALZ

		// There is no outgoing edge starting with
		// activeEdge from activeNode
		if (activeNode->children] == NULL)
		{
			//Extension Rule 2 (A new leaf edge gets created)
			activeNode->children] =
										newNode(pos, &leafEnd);

			/*A new leaf edge is created in above line starting
			from an existng node (the current activeNode), and
			if there is any internal node waiting for it's suffix
			link get reset, point the suffix link from that last
			internal node to current activeNode. Then set lastNewNode
			to NULL indicating no more node waiting for suffix link
			reset.*/
			if (lastNewNode != NULL)
			{
				lastNewNode->suffixLink = activeNode;
				lastNewNode = NULL;
			}
		}
		// There is an outgoing edge starting with activeEdge
		// from activeNode
		else
		{
			// Get the next node at the end of edge starting
			// with activeEdge
			Node *next = activeNode->children] ;
			if (walkDown(next))//Do walkdown
			{
				//Start from next node (the new activeNode)
				continue;
			}
			/*Extension Rule 3 (current character being processed
			is already on the edge)*/
			if (text[next->start + activeLength] == text[pos])
			{
				//APCFER3
				activeLength++;
				/*STOP all further processing in this phase
				and move on to next phase*/
				break;
			}

			/*We will be here when activePoint is in middle of
			the edge being traversed and current character
			being processed is not on the edge (we fall off
			the tree). In this case, we add a new internal node
			and a new leaf edge going out of that new node.*/
			splitEnd = (int*) malloc(sizeof(int));
			*splitEnd = next->start + activeLength - 1;

			//New internal node
			Node *split = newNode(next->start, splitEnd);
			activeNode->children] = split;

			//New leaf coming out of new internal node
			split->children] = newNode(pos, &leafEnd);
			next->start += activeLength;
			split->children] = next;

			if (lastNewNode != NULL)
			{
			/*suffixLink of lastNewNode points to current newly
			created internal node*/
				lastNewNode->suffixLink = split;
			}

			lastNewNode = split;
		}

		/* One suffix got added in tree, decrement the count of
		suffixes yet to be added.*/
		remainingSuffixCount--;
		if (activeNode == root && activeLength > 0)
		{
			activeLength--;
			activeEdge = pos - remainingSuffixCount + 1;
		}
		else if (activeNode != root)
		{
			activeNode = activeNode->suffixLink;
		}
	}
}

void print(int i, int j)
{
	int k;
	for (k=i; k<=j && text[k] != '#'; k++)
		printf("%c", text[k]);
	if(k<=j)
		printf("#");
}

void setSuffixIndexByDFS(Node *n, int labelHeight)
{
	if (n == NULL) return;

	if (n->start != -1) //A non-root node
	{
		//Print the label on edge from parent to current node
		//Uncomment below line to print suffix tree
		//print(n->start, *(n->end));
	}
	int leaf = 1;
	int i;
	for (i = 0; i < MAX_CHAR; i++)
	{
		if (n->children[i] != NULL)
		{

			//Current node is not a leaf as it has outgoing
			//edges from it.
			leaf = 0;
			setSuffixIndexByDFS(n->children[i], labelHeight +
								edgeLength(n->children[i]));
			if(n != root)
			{
				//Add chldren's suffix indices in parent
				n->forwardIndices->insert(
					n->children[i]->forwardIndices->begin(),
					n->children[i]->forwardIndices->end());
				n->reverseIndices->insert(
					n->children[i]->reverseIndices->begin(),
					n->children[i]->reverseIndices->end());
			}
		}
	}
	if (leaf == 1)
	{
		for(i= n->start; i<= *(n->end); i++)
		{
			if(text[i] == '#')
			{
				n->end = (int*) malloc(sizeof(int));
				*(n->end) = i;
			}
		}
		n->suffixIndex = size - labelHeight;

		if(n->suffixIndex < size1) //Suffix of Given String
			n->forwardIndices->insert(n->suffixIndex);
		else //Suffix of Reversed String
			n->reverseIndices->insert(n->suffixIndex - size1);

		//Uncomment below line to print suffix index
	// printf(" [%d]\n", n->suffixIndex);
	}
}

void freeSuffixTreeByPostOrder(Node *n)
{
	if (n == NULL)
		return;
	int i;
	for (i = 0; i < MAX_CHAR; i++)
	{
		if (n->children[i] != NULL)
		{
			freeSuffixTreeByPostOrder(n->children[i]);
		}
	}
	if (n->suffixIndex == -1)
		free(n->end);
	free(n);
}

/*Build the suffix tree and print the edge labels along with suffixIndex.
suffixIndex for leaf edges will be >= 0 and
for non-leaf edges will be -1*/
void buildSuffixTree()
{
	size = strlen(text);
	int i;
	rootEnd = (int*) malloc(sizeof(int));
	*rootEnd = - 1;

	/*Root is a special node with start and end indices as -1,
	as it has no parent from where an edge comes to root*/
	root = newNode(-1, rootEnd);

	activeNode = root; //First activeNode will be root
	for (i=0; i<size; i++)
		extendSuffixTree(i);
	int labelHeight = 0;
	setSuffixIndexByDFS(root, labelHeight);
}

void doTraversal(Node *n, int labelHeight, int* maxHeight,
int* substringStartIndex)
{
	if(n == NULL)
	{
		return;
	}
	int i=0;
	int ret = -1;
	if(n->suffixIndex < 0) //If it is internal node
	{
		for (i = 0; i < MAX_CHAR; i++)
		{
			if(n->children[i] != NULL)
			{
				doTraversal(n->children[i], labelHeight +
					edgeLength(n->children[i]),
					maxHeight, substringStartIndex);

				if(*maxHeight < labelHeight
					&& n->forwardIndices->size() > 0 &&
					n->reverseIndices->size() > 0)
				{
					for (forwardIndex=n->forwardIndices->begin();
							forwardIndex!=n->forwardIndices->end();
							++forwardIndex)
					{
						reverseIndex = (size1 - 2) -
							(*forwardIndex + labelHeight - 1);
						//If reverse suffix comes from
						//SAME position in given string
						//Keep track of deepest node
						if(n->reverseIndices->find(reverseIndex) !=
							n->reverseIndices->end())
						{
							*maxHeight = labelHeight;
							*substringStartIndex = *(n->end) -
								labelHeight + 1;
							break;
						}
					}
				}
			}
		}
	}
}

void getLongestPalindromicSubstring()
{
	int maxHeight = 0;
	int substringStartIndex = 0;
	doTraversal(root, 0, &maxHeight, &substringStartIndex);

	int k;
	for (k=0; k<maxHeight; k++)
		printf("%c", text[k + substringStartIndex]);
	if(k == 0)
		printf("No palindromic substring");
	else
		printf(", of length: %d",maxHeight);
	printf("\n");
}

// driver program to test above functions
int main(int argc, char *argv[])
{
	size1 = 49;
	printf("Longest Palindromic Substring in ottorotorrentnertrugtimeinesohellehoseniemitgurt is: ");
	strcpy(text, "ottorotorrentnertrugtimeinesohellehoseniemitgurt#trugtimeinesohellehoseniemitgurtrentnerrotorotto$"); buildSuffixTree();
	getLongestPalindromicSubstring();
	//Free the dynamically allocated memory
	freeSuffixTreeByPostOrder(root);


	return 0;
}