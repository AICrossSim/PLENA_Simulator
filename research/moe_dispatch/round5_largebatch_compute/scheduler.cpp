// Deterministic whole-job unrelated-machine scheduling bounds and incumbents.
// Every returned upper bound is a complete legal nonpreemptive assignment.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <cstdint>
using std::vector;

double lower(const double* p,int n,int nc,const int* typ){
  double individual=0;for(int i=0;i<n;i++){double v=1e100;for(int c=0;c<nc;c++)v=std::min(v,p[i*nc+c]);individual=std::max(individual,v);}
  int a=0,b=-1,na=0,nb=0;
  for(int c=0;c<nc;c++){if(typ[c]==typ[0])na++;else{b=c;nb++;}}
  if(b<0){double s=0;for(int i=0;i<n;i++)s+=p[i*nc];return std::max(individual,s/nc);}
  // Fractional two-pool relaxation: each pool can distribute its work freely.
  // Sorting by a_i/b_i gives the continuous knapsack frontier; interpolate
  // the unique point where the two normalized pool loads cross.
  vector<int> order(n);std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](int i,int j){return p[i*nc+a]*p[j*nc+b]<p[j*nc+a]*p[i*nc+b];});
  auto fractional=[&](int fixed,int side){
    double la=0,lb=0;
    if(fixed>=0&&side==0)la=p[fixed*nc+a]/na;
    for(int i=0;i<n;i++)if(i!=fixed||side==1)lb+=p[i*nc+b]/nb;
    if(la>=lb)return la;
    for(int i:order){if(i==fixed)continue;double da=p[i*nc+a]/na,db=p[i*nc+b]/nb;
      if(la+da>=lb-db){double x=(lb-la)/(da+db);return la+x*da;}
      la+=da;lb-=db;
    }
    return std::max(la,lb);
  };
  double basic=std::max(individual,fractional(-1,-1));
  int largest=0;for(int i=1;i<n;i++)if(std::min(p[i*nc+a],p[i*nc+b])>std::min(p[largest*nc+a],p[largest*nc+b]))largest=i;
  // The largest job must wholly choose one pool. Min across these exhaustive
  // branches is still a valid global lower bound, with a stronger pmax term.
  double branchA=std::max(p[largest*nc+a],fractional(largest,0));
  double branchB=std::max(p[largest*nc+b],fractional(largest,1));
  return std::max(basic,std::min(branchA,branchB));
}

double solve(const double* p,int n,int nc,int* owner,int style,bool improve=true){
 vector<int> order(n);std::iota(order.begin(),order.end(),0);
 // Each key is immutable during this sort. Cache it once instead of
 // rescanning every core at every comparator call; ordering is identical.
 vector<double> keys(n);
 for(int i=0;i<n;i++){double mn=1e100,mx=0;for(int c=0;c<nc;c++){mn=std::min(mn,p[i*nc+c]);mx=std::max(mx,p[i*nc+c]);}
   keys[i]=style==0?mn:style==1?mx:mx-mn;}
 std::stable_sort(order.begin(),order.end(),[&](int i,int j){return keys[i]>keys[j];});
 vector<double> loads(nc,0);
 for(int i:order){int c=0;for(int k=1;k<nc;k++)if(loads[k]+p[i*nc+k]<loads[c]+p[i*nc+c])c=k;owner[i]=c;loads[c]+=p[i*nc+c];}
 // Monotone local moves and swaps. Positive termination criterion prevents
 // cycles, and load recomputation below validates each returned assignment.
 for(int pass=0;pass<(improve?12:0);pass++){
  double old=*std::max_element(loads.begin(),loads.end()),best=old;int bi=-1,bj=-1,bc=-1;
  int heavy=std::max_element(loads.begin(),loads.end())-loads.begin();
  for(int i=0;i<n;i++)if(owner[i]==heavy){
   for(int c=0;c<nc;c++)if(c!=heavy){double t=0;
    for(int k=0;k<nc;k++)t=std::max(t,loads[k]+(k==heavy?-p[i*nc+heavy]:k==c?p[i*nc+c]:0));
    if(t<best-1e-8){best=t;bi=i;bj=-1;bc=c;}
   }
   for(int j=0;j<n;j++)if(owner[j]!=heavy){int c=owner[j];double t=0;
    for(int k=0;k<nc;k++)t=std::max(t,loads[k]+(k==heavy?-p[i*nc+heavy]+p[j*nc+heavy]:k==c?-p[j*nc+c]+p[i*nc+c]:0));
    if(t<best-1e-8){best=t;bi=i;bj=j;bc=c;}
   }
  }
  if(bi<0)break;
  loads[heavy]-=p[bi*nc+heavy];loads[bc]+=p[bi*nc+bc];owner[bi]=bc;
  if(bj>=0){loads[bc]-=p[bj*nc+bc];loads[heavy]+=p[bj*nc+heavy];owner[bj]=heavy;}
 }
 return *std::max_element(loads.begin(),loads.end());
}

extern "C" void schedule(const double* p,int n,int nc,const int* types,double* bounds,int* owner){
 bounds[0]=lower(p,n,nc,types);bounds[1]=1e100;vector<int> tmp(n);
 for(int style=0;style<3;style++){double ub=solve(p,n,nc,tmp.data(),style);if(ub<bounds[1]){bounds[1]=ub;std::copy(tmp.begin(),tmp.end(),owner);}}
}

extern "C" void grid(const double* costs,int nw,int nt,int ns,const int* counts,
                      const int* cfg,int ncfg,int maxcore,double* lbs,double* ubs){
 #pragma omp parallel for schedule(dynamic,8)
 for(int d=0;d<ncfg;d++){
  vector<double> p(nt*maxcore);vector<int> own(nt),types(maxcore);
  int nc=0;while(nc<maxcore&&cfg[d*maxcore+nc]>=0)nc++;
  for(int c=0;c<nc;c++)types[c]=cfg[d*maxcore+c];
  for(int w=0;w<nw;w++){
   int n=counts[w];for(int i=0;i<n;i++)for(int c=0;c<nc;c++)p[i*nc+c]=costs[(types[c]*nw+w)*nt+i];
   lbs[d*nw+w]=lower(p.data(),n,nc,types.data());double ub=1e100;
   for(int style=0;style<2;style++)ub=std::min(ub,solve(p.data(),n,nc,own.data(),style,false));
   ubs[d*nw+w]=ub;
  }
 }
}
