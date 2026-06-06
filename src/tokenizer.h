// Self-contained GPT-2 byte-level BPE tokenizer for LFM2.5 (matches tokenizer.json).
// Loads vocab.tsv / merges.tsv / special.tsv exported by tools/export_tokenizer.py.
// Author: g023 (https://github.com/g023/)
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

struct Tokenizer {
    std::unordered_map<std::string,int> vocab;     // byte-level token string -> id
    std::vector<std::string> id2tok;               // id -> byte-level string (base vocab)
    std::unordered_map<std::string,int> mergerank; // "A\x01B" -> rank
    std::vector<std::pair<std::string,int>> specials; // (content, id), longest-first
    std::unordered_map<int,std::string> id2special;
    std::string b2u[256];                          // byte -> utf8 of byte-level unicode char
    std::unordered_map<std::string,int> u2b;       // byte-level unicode char (utf8) -> byte

    static void cp_to_utf8(uint32_t cp, std::string& out){
        if(cp<0x80) out.push_back((char)cp);
        else if(cp<0x800){ out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
        else if(cp<0x10000){ out.push_back((char)(0xE0|(cp>>12))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
        else { out.push_back((char)(0xF0|(cp>>18))); out.push_back((char)(0x80|((cp>>12)&0x3F))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
    }
    void build_byte_maps(){
        std::vector<int> bs, cs;
        for(int b=33;b<=126;b++){bs.push_back(b);cs.push_back(b);}
        for(int b=161;b<=172;b++){bs.push_back(b);cs.push_back(b);}
        for(int b=174;b<=255;b++){bs.push_back(b);cs.push_back(b);}
        int n=0;
        std::vector<int> map(256,-1);
        for(size_t i=0;i<bs.size();i++) map[bs[i]]=cs[i];
        for(int b=0;b<256;b++) if(map[b]<0){ map[b]=256+n; n++; }
        for(int b=0;b<256;b++){ std::string s; cp_to_utf8((uint32_t)map[b], s); b2u[b]=s; u2b[s]=b; }
    }

    bool load(const std::string& dir){
        build_byte_maps();
        // vocab
        FILE* f=fopen((dir+"/vocab.tsv").c_str(),"r"); if(!f){perror("vocab.tsv");return false;}
        char* line=nullptr; size_t cap=0; ssize_t len;
        int maxid=0;
        std::vector<std::pair<int,std::string>> tmp;
        while((len=getline(&line,&cap,f))>=0){
            if(len>0 && line[len-1]=='\n'){ line[--len]='\0'; }
            char* tab=strchr(line,'\t'); if(!tab) continue;
            *tab='\0'; int id=atoi(line); std::string tokstr(tab+1);
            tmp.push_back({id,tokstr}); vocab[tokstr]=id; if(id>maxid)maxid=id;
        }
        fclose(f);
        id2tok.assign(maxid+1,"");
        for(auto&p:tmp) id2tok[p.first]=p.second;
        // merges
        f=fopen((dir+"/merges.tsv").c_str(),"r"); if(!f){perror("merges.tsv");return false;}
        while((len=getline(&line,&cap,f))>=0){
            if(len>0 && line[len-1]=='\n'){ line[--len]='\0'; }
            char* t1=strchr(line,'\t'); if(!t1) continue; *t1='\0'; int rank=atoi(line);
            char* a=t1+1; char* t2=strchr(a,'\t'); if(!t2) continue; *t2='\0'; char* b=t2+1;
            std::string key=std::string(a)+std::string("\x01")+std::string(b);
            mergerank[key]=rank;
        }
        fclose(f);
        // special
        f=fopen((dir+"/special.tsv").c_str(),"r"); if(!f){perror("special.tsv");return false;}
        while((len=getline(&line,&cap,f))>=0){
            if(len>0 && line[len-1]=='\n'){ line[--len]='\0'; }
            char* tab=strchr(line,'\t'); if(!tab) continue; *tab='\0'; int id=atoi(line);
            std::string content(tab+1);
            specials.push_back({content,id}); id2special[id]=content;
        }
        fclose(f);
        std::sort(specials.begin(),specials.end(),[](auto&a,auto&b){return a.first.size()>b.first.size();});
        free(line);
        return true;
    }

    // --- utf8 decode to codepoints ---
    static std::vector<uint32_t> to_cps(const std::string& s){
        std::vector<uint32_t> cps; size_t i=0, n=s.size();
        while(i<n){ unsigned char c=s[i];
            uint32_t cp; int k;
            if(c<0x80){cp=c;k=1;} else if((c>>5)==0x6){cp=c&0x1F;k=2;} else if((c>>4)==0xE){cp=c&0xF;k=3;} else {cp=c&0x7;k=4;}
            for(int j=1;j<k && i+j<n;j++) cp=(cp<<6)|(s[i+j]&0x3F);
            cps.push_back(cp); i+=k;
        }
        return cps;
    }
    static bool is_ws(uint32_t c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c==0x0b||c==0x0c||c==0xa0; }
    static bool is_digit(uint32_t c){ return c>='0'&&c<='9'; }
    static bool is_letter(uint32_t c){
        if((c>='A'&&c<='Z')||(c>='a'&&c<='z')) return true;
        if(c>=0x80 && !is_ws(c)) return true;   // approximate \p{L} for non-ascii
        return false;
    }
    static bool is_punct(uint32_t c){ return !is_ws(c)&&!is_letter(c)&&!is_digit(c); }
    static bool is_lower(uint32_t c){ return c>='a'&&c<='z'; }
    static uint32_t lower(uint32_t c){ return (c>='A'&&c<='Z')? c+32:c; }

    // GPT-2 pretokenizer: returns piece [start,end) codepoint ranges.
    std::vector<std::pair<int,int>> pretok(const std::vector<uint32_t>& c){
        int n=c.size(); std::vector<std::pair<int,int>> out; int i=0;
        auto contraction=[&](int i)->int{
            if(c[i]!='\'') return 0; if(i+1>=n) return 0;
            uint32_t a=lower(c[i+1]);
            if(a=='s'||a=='t'||a=='m'||a=='d') return 2;
            if(i+2<n){ uint32_t b=lower(c[i+2]);
                if((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')) return 3; }
            return 0;
        };
        while(i<n){
            int m;
            // A contraction
            if((m=contraction(i))){ out.push_back({i,i+m}); i+=m; continue; }
            // B optional non-LN then letters
            if(is_letter(c[i])){ int j=i; while(j<n&&is_letter(c[j]))j++; out.push_back({i,j}); i=j; continue; }
            if(!is_digit(c[i]) && c[i]!='\r' && c[i]!='\n' && !is_ws(c[i]) && i+1<n && is_letter(c[i+1])){
                int j=i+1; while(j<n&&is_letter(c[j]))j++; out.push_back({i,j}); i=j; continue; }
            // also: optional space (which is ws) then letters -> handled here since space allowed as the single optional [^\r\n\p{L}\p{N}]
            if(c[i]==' ' && i+1<n && is_letter(c[i+1])){ int j=i+1; while(j<n&&is_letter(c[j]))j++; out.push_back({i,j}); i=j; continue; }
            // C digits 1-3
            if(is_digit(c[i])){ int j=i; int cnt=0; while(j<n&&is_digit(c[j])&&cnt<3){j++;cnt++;} out.push_back({i,j}); i=j; continue; }
            // D optional space then punct+ then \r\n*
            {
                int j=i; bool sp=false;
                if(c[j]==' '){ sp=true; }
                int p = sp? j+1 : j;
                if(p<n && is_punct(c[p])){
                    int q=p; while(q<n&&is_punct(c[q]))q++;
                    while(q<n&&(c[q]=='\r'||c[q]=='\n'))q++;
                    out.push_back({i,q}); i=q; continue;
                }
            }
            // E \s*[\r\n]+
            if(is_ws(c[i])){
                int e=i; while(e<n&&is_ws(c[e]))e++;
                int lastnl=-1; for(int k=i;k<e;k++) if(c[k]=='\r'||c[k]=='\n') lastnl=k;
                if(lastnl>=0){ out.push_back({i,lastnl+1}); i=lastnl+1; continue; }
                // F \s+(?!\S)
                if(e==n){ out.push_back({i,e}); i=e; continue; }
                if(e-1>i){ out.push_back({i,e-1}); i=e-1; continue; }
                // G single ws followed by non-space
                out.push_back({i,e}); i=e; continue;
            }
            // fallback single char
            out.push_back({i,i+1}); i++;
        }
        return out;
    }

    void bpe_word(const std::string& word, std::vector<int>& out){
        // word: byte-level unicode string. If whole in vocab (ignore_merges), emit directly.
        auto it=vocab.find(word);
        if(it!=vocab.end()){ out.push_back(it->second); return; }
        // split into utf8 chars
        std::vector<std::string> syms; size_t i=0,n=word.size();
        while(i<n){ unsigned char ch=word[i]; int k=(ch<0x80)?1:((ch>>5)==0x6?2:((ch>>4)==0xE?3:4));
            syms.push_back(word.substr(i,k)); i+=k; }
        while(syms.size()>1){
            int bestRank=INT32_MAX, bestI=-1;
            for(size_t j=0;j+1<syms.size();j++){
                auto m=mergerank.find(syms[j]+std::string("\x01")+syms[j+1]);
                if(m!=mergerank.end() && m->second<bestRank){ bestRank=m->second; bestI=j; }
            }
            if(bestI<0) break;
            syms[bestI]=syms[bestI]+syms[bestI+1];
            syms.erase(syms.begin()+bestI+1);
        }
        for(auto&s:syms){ auto v=vocab.find(s); if(v!=vocab.end()) out.push_back(v->second);
            else { /* should not happen: every byte-level char is in vocab */ } }
    }

    std::vector<int> encode(const std::string& text){
        std::vector<int> out;
        // split off special tokens (literal match, longest first)
        size_t pos=0, n=text.size();
        while(pos<n){
            // find earliest special occurrence
            size_t bestPos=std::string::npos; int bestId=-1; size_t bestLen=0;
            for(auto&sp:specials){
                size_t f=text.find(sp.first,pos);
                if(f!=std::string::npos && (f<bestPos || (f==bestPos && sp.first.size()>bestLen))){
                    bestPos=f; bestId=sp.second; bestLen=sp.first.size();
                }
            }
            size_t chunkEnd = (bestPos==std::string::npos)? n : bestPos;
            if(chunkEnd>pos){
                std::string chunk=text.substr(pos,chunkEnd-pos);
                auto cps=to_cps(chunk);
                auto pieces=pretok(cps);
                // need codepoint -> byte offset to extract raw bytes per piece; rebuild via cp utf8
                // build byte string per piece from codepoints
                for(auto&pc:pieces){
                    std::string raw;
                    for(int k=pc.first;k<pc.second;k++) cp_to_utf8(cps[k], raw);
                    // byte-level encode
                    std::string bl; for(unsigned char b: raw) bl+=b2u[b];
                    bpe_word(bl, out);
                }
            }
            if(bestPos==std::string::npos) break;
            out.push_back(bestId); pos=bestPos+bestLen;
        }
        return out;
    }

    // map a byte-level unicode string back to raw bytes
    std::string bl_to_raw(const std::string& bl){
        std::string raw; size_t i=0;
        while(i<bl.size()){ unsigned char ch=bl[i]; int k=(ch<0x80)?1:((ch>>5)==0x6?2:((ch>>4)==0xE?3:4));
            auto it=u2b.find(bl.substr(i,k)); if(it!=u2b.end()) raw.push_back((char)it->second); i+=k; }
        return raw;
    }
    std::string decode(const std::vector<int>& ids, bool skip_special=false){
        std::string out, bl;
        for(int id: ids){
            auto s=id2special.find(id);
            if(s!=id2special.end()){
                out += bl_to_raw(bl); bl.clear();
                if(!skip_special) out += s->second;
                continue;
            }
            if(id>=0 && id<(int)id2tok.size()) bl += id2tok[id];
        }
        out += bl_to_raw(bl);
        return out;
    }
};
