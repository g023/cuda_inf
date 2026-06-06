#include "tokenizer.h"
#include <cstdio>
int main(int argc, char** argv){
    Tokenizer tk;
    const char* tokdir = argc > 1 ? argv[1] : "scratch/engine_weights/tok";
    if(!tk.load(tokdir)) return 1;
    const char* tests[] = {
        "<|startoftext|><|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n",
        "The capital of France is Paris.",
        "Hello, world! 123 numbers and  multiple   spaces.\nNew line.",
        "def foo(x): return x*2  # comment",
    };
    for(auto t: tests){
        auto ids=tk.encode(t);
        printf("TEXT: %s\nIDS:", t);
        for(int id:ids) printf(" %d", id);
        printf("\nDEC: %s\n---\n", tk.decode(ids).c_str());
    }
    return 0;
}
