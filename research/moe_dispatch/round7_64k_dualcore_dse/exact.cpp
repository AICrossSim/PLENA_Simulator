// Exact two-unrelated-machine assignment via Pareto dynamic programming.
// State (load0, saved_load1, mask). Lower load0 and higher saved_load1 dominate.
// Group identical (p0,p1), binary-decompose counts. No heuristic/time cutoff.
#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <utility>
#include <vector>
using I = int64_t;
using M = __uint128_t;
struct State { I a,b; M mask; };
extern "C" I exact_assign(int n,const I* a,const I* b,uint64_t* mask_out,I* details) {
    if(n<1 || n>66) return -1;
    I total=std::accumulate(b,b+n,I(0));
    std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int i,int j){return std::min(a[i],b[i])>std::min(a[j],b[j]);});
    I l0=0,l1=0;
    for(int i:order) {if(l0+a[i]<=l1+b[i])l0+=a[i];else l1+=b[i];}
    I bound=std::max(l0,l1),peak=1;
    std::map<std::pair<I,I>,std::vector<int>> groups;
    for(int i=0;i<n;++i){if(a[i]<=0||b[i]<=0)return -2;groups[{a[i],b[i]}].push_back(i);}
    std::vector<State> dp{{0,0,0}};
    for(auto &g:groups){
        size_t used=0;int step=1;
        while(used<g.second.size()){
            int take=std::min<size_t>(step,g.second.size()-used);M mask=0;
            for(int j=0;j<take;++j)mask|=M(1)<<g.second[used+j];
            I da=g.first.first*take,db=g.first.second*take;
            std::vector<State> next;next.reserve(dp.size()*2);
            size_t i=0,j=0;I best_saved=-1;
            while(i<dp.size() || (j<dp.size() && dp[j].a+da<=bound)){
                bool old=i<dp.size();bool shifted=j<dp.size() && dp[j].a+da<=bound;
                State z;
                if(old && (!shifted || dp[i].a<dp[j].a+da))z=dp[i++];
                else if(shifted && (!old || dp[j].a+da<dp[i].a)){
                    z={dp[j].a+da,dp[j].b+db,dp[j].mask|mask};++j;
                }else{
                    State x=dp[i++],y={dp[j].a+da,dp[j].b+db,dp[j].mask|mask};++j;
                    z=(x.b>y.b || (x.b==y.b && x.mask<y.mask))?x:y;
                }
                if(z.b>best_saved){next.push_back(z);best_saved=z.b;}
            }
            dp.swap(next);peak=std::max<I>(peak,dp.size());used+=take;step*=2;
        }
    }
    I optimum=INT64_MAX;State chosen{};
    for(auto &s:dp){I t=std::max(s.a,total-s.b);
        if(t<optimum || (t==optimum && s.a<chosen.a)){optimum=t;chosen=s;}}
    mask_out[0]=uint64_t(chosen.mask);mask_out[1]=uint64_t(chosen.mask>>64);
    details[0]=chosen.a;details[1]=total-chosen.b;details[2]=peak;
    return optimum;
}
