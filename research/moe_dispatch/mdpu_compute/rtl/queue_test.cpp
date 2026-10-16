#include "Vmdpu_queue_engine.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct Task {unsigned id,m,shared,c[2];};
struct Shadow {bool valid=false;Task task{};int claim=0;int setup_words=0;};
struct Result {unsigned cycles=0,accepted=0,issued=0,backpressure=0,steals=0,dual_issues=0;};
void check(bool ok,const std::string& why) {if(!ok)throw std::runtime_error(why);}
Result run(unsigned seed,bool bypass,bool prefetch,bool long_dense=false) {
    Vmdpu_queue_engine d; std::mt19937 rng(seed);
    d.clk=0;d.reset=1;d.in_valid=0;d.eval();d.clk=1;d.eval();d.clk=0;d.reset=0;
    d.threshold=2;d.bypass_enable=bypass;d.head_bypass_enable=bypass;d.prefetch_enable=prefetch;
    d.wait_best=0;
    std::deque<Task> q[2]; Shadow shadow[2];
    std::array<unsigned,2> busy{0,0};
    std::array<bool,96> accepted{},issued{};
    std::vector<Task> source;
    for(unsigned i=0;i<96;++i) source.push_back({i,unsigned(1+rng()%16),unsigned(i==0),
        {long_dense?32u:unsigned(1+rng()%55),long_dense?128u:unsigned(1+rng()%77)}});
    if(long_dense)for(auto&t:source)t.shared=1;
    unsigned pos=0; Result result;
    for(unsigned t=0;t<20000;++t) {
        unsigned drains[2]={busy[0],busy[1]};
        for(int c=0;c<2;++c) {
            if(shadow[c].valid)drains[c]+=shadow[c].task.c[c]+4;
            for(auto& x:q[c])drains[c]+=x.c[c]+4;
        }
        d.drain0=drains[0];d.drain1=drains[1];
        d.current_remaining0=busy[0];d.current_remaining1=busy[1];
        d.core_ready0=busy[0]==0;d.core_ready1=busy[1]==0;
        // Each descriptor remains stable while the receiver is backpressured.
        d.in_valid=pos<source.size();
        if(d.in_valid) {auto&x=source[pos];d.in_expert=x.id;d.in_tokens=x.m;
            d.in_shared=x.shared;d.in_cost0=x.c[0];d.in_cost1=x.c[1];}
        d.clk=0;d.eval();
        check(d.count0==q[0].size() && d.count1==q[1].size(),"FIFO occupancy mismatch");
        check(d.shadow_valid0==shadow[0].valid && d.shadow_valid1==shadow[1].valid,"shadow valid mismatch");
        bool fires[2]={bool(d.issue_valid0&&d.core_ready0),bool(d.issue_valid1&&d.core_ready1)};
        unsigned ids[2]={d.issue_expert0,d.issue_expert1};
        unsigned costs[2]={d.issue_cost0,d.issue_cost1};
        bool claims[2]={bool(d.claim_valid0),bool(d.claim_valid1)};
        bool steals[2]={bool(d.stolen0),bool(d.stolen1)};
        if(fires[0]&&fires[1])++result.dual_issues;
        for(int c=0;c<2;++c) {
            if(fires[c]) {
                check(shadow[c].valid,"empty shadow granted");
                check(shadow[c].setup_words>=4,"context issued before four setup words");
                check(t>=unsigned(shadow[c].claim+4),"context setup latency shorter than four cycles");
                check(ids[c]==shadow[c].task.id && costs[c]==shadow[c].task.c[c],"descriptor corruption");
                check(accepted[ids[c]] && !issued[ids[c]],"unaccepted or duplicate issue");
                issued[ids[c]]=true;++result.issued;shadow[c].valid=false;
            }
        }
        for(int c=0;c<2;++c) {
            if(shadow[c].valid && (prefetch || busy[c]==0))++shadow[c].setup_words;
            if(claims[c]) {
                unsigned src=steals[c]?1-c:c;
                check(!shadow[c].valid,"claim overwrites occupied shadow");
                check(!q[src].empty(),"empty FIFO granted");
                if(steals[c]){check(bypass,"steal when bypass disabled");++result.steals;}
                shadow[c]={true,q[src].front(),int(t),int(prefetch||busy[c]==0)};
                q[src].pop_front();
            }
        }
        if(d.in_valid && d.in_ready) {
            unsigned dest=d.accepted_stream;
            check(q[dest].size()<8,"FIFO overflow");
            if(!bypass)check(dest==unsigned(!source[pos].shared && source[pos].m<=2),"hard affinity violated");
            q[dest].push_back(source[pos]);accepted[pos]=true;++pos;++result.accepted;
        } else if(d.in_valid) ++result.backpressure;
        for(int c=0;c<2;++c) {if(busy[c])--busy[c];if(fires[c])busy[c]=costs[c]-1;}
        d.clk=1;d.eval();
        result.cycles=t+1;
        if(result.issued==96 && !busy[0] && !busy[1])break;
    }
    check(result.issued==96 && result.accepted==96,"lost tasks or controller deadlock");
    check(q[0].empty()&&q[1].empty(),"tasks left behind after drain");
    return result;
}
unsigned replay(const char* file){
    std::ifstream in(file);check(bool(in),"cannot open runtime replay vectors");
    std::string name;unsigned n,bypass,head_bypass,prefetch,wait_best,threshold,expected_end,cases=0;
    while(in>>name>>n>>bypass>>head_bypass>>prefetch>>wait_best>>threshold>>expected_end){
        std::vector<Task> jobs(n);std::array<unsigned,128> owners{},starts{},ends{},qowners{},qentry{},claimclock{};
        for(auto&x:jobs){unsigned owner,start,end;
            in>>x.id>>x.m>>x.shared>>x.c[0]>>x.c[1]>>owner>>start>>end;
            in>>qowners[x.id]>>qentry[x.id]>>claimclock[x.id];
            owners[x.id]=owner;starts[x.id]=start;ends[x.id]=end;
        }
        Vmdpu_queue_engine d;d.clk=0;d.reset=1;d.eval();d.clk=1;d.eval();d.clk=0;d.reset=0;
        d.threshold=threshold;d.bypass_enable=bypass;d.head_bypass_enable=head_bypass;d.prefetch_enable=prefetch;
        d.wait_best=wait_best;
        std::array<unsigned,2> busy{0,0};unsigned pos=0,done=0,actual_end=0;
        for(unsigned t=0;t<expected_end+100000;++t){
            d.current_remaining0=busy[0];d.current_remaining1=busy[1];
            d.core_ready0=busy[0]==0;d.core_ready1=busy[1]==0;
            d.drain0=0;d.drain1=0;d.in_valid=pos<n;
            if(pos<n){auto&x=jobs[pos];d.in_expert=x.id;d.in_tokens=x.m;d.in_shared=x.shared;d.in_cost0=x.c[0];d.in_cost1=x.c[1];}
            d.clk=0;d.eval();
            bool fire[2]={bool(d.issue_valid0&&d.core_ready0),bool(d.issue_valid1&&d.core_ready1)};
            unsigned eid[2]={d.issue_expert0,d.issue_expert1},cost[2]={d.issue_cost0,d.issue_cost1};
            for(unsigned c=0;c<2;++c)if(fire[c]){
                if(owners[eid[c]]!=c||starts[eid[c]]!=t||ends[eid[c]]!=t+cost[c]){
                    std::cerr<<name<<" expert="<<eid[c]<<" actual="<<c<<","<<t<<","<<t+cost[c]
                        <<" expected="<<owners[eid[c]]<<","<<starts[eid[c]]<<","<<ends[eid[c]]<<"\n";
                    throw std::runtime_error("full runtime replay issue mismatch");
                }
                ++done;actual_end=std::max(actual_end,t+cost[c]);
            }
            if(d.in_valid&&d.in_ready){
                unsigned id=jobs[pos].id;
                if(d.accepted_stream!=qowners[id]||t!=qentry[id]){
                    std::cerr<<name<<" ingress expert="<<id<<" actual="<<unsigned(d.accepted_stream)<<","<<t
                        <<" expected="<<qowners[id]<<","<<qentry[id]<<"\n";
                    throw std::runtime_error("runtime replay ingress mismatch");
                }
                ++pos;
            }
            for(unsigned c=0;c<2;++c){if(busy[c])--busy[c];if(fire[c])busy[c]=cost[c]-1;}
            d.clk=1;d.eval();
            if(done==n)break;
        }
        check(done==n&&actual_end==expected_end,name+" makespan mismatch");++cases;
    }
    return cases;
}
int main(int argc,char**argv){
    Result totals;unsigned runs=0;
    for(unsigned seed=0;seed<64;++seed)for(bool bypass:{false,true})for(bool prefetch:{false,true}) {
        auto r=run(seed,bypass,prefetch);++runs;
        totals.cycles+=r.cycles;totals.accepted+=r.accepted;totals.issued+=r.issued;
        totals.backpressure+=r.backpressure;totals.steals+=r.steals;totals.dual_issues+=r.dual_issues;
    }
    auto off=run(11,false,false,true),on=run(11,false,true,true);
    check(on.cycles<off.cycles,"prefetch failed to hide independent context setup");
    check(totals.steals>0 && totals.dual_issues>0 && totals.backpressure>0,"missing exercised corner case");
    unsigned replays=argc>1?replay(argv[1]):0;
    std::cout<<"{\"queue_scenarios\":"<<runs+2<<",\"runtime_trace_replays\":"<<replays<<",\"random_task_instances\":"<<totals.issued
      <<",\"cycles_checked\":"<<totals.cycles<<",\"backpressure_cycles\":"<<totals.backpressure
      <<",\"head_steals\":"<<totals.steals<<",\"dual_issue_cycles\":"<<totals.dual_issues
      <<",\"context_prefetch_off_cycles\":"<<off.cycles<<",\"context_prefetch_on_cycles\":"<<on.cycles
      <<",\"passed\":true}\n";
}
