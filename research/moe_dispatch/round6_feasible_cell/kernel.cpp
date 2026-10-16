#include "../round5_largebatch_compute/scheduler.cpp"

double phase6(double C,int blocks,double d){return std::max(C+d,blocks*d+std::floor(C/blocks));}
double cohort6(const vector<int>& sc,const double* gu,const double* dn,double bw){
 int k=0;double cg=0,cd=0;
 for(int c=0;c<(int)sc.size();c++){k+=sc[c]>0;cg=std::max(cg,sc[c]*gu[c]);cd=std::max(cd,sc[c]*dn[c]);}
 if(!k)return 0;
 double d=bw>0?std::ceil(262144.0*(k-1)/(bw*1000)):0;
 return phase6(cg,88,d)+phase6(cd,44,d);
}
double evaluate6(const double* p,int n,int nc,int ns,const double* gu,const double* dn,const int* own,double bw){
 vector<int> sc(nc,0);vector<double> r(nc,0);
 for(int i=0;i<n;i++){if(i>=n-ns)sc[own[i]]++;else r[own[i]]+=p[i*nc+own[i]];}
 double bc=cohort6(sc,gu,dn,bw),t=0;
 for(int c=0;c<nc;c++)t=std::max(t,r[c]+(sc[c]?bc:0));return t;
}
double upper6(const double* p,int n,int nc,int ns,const double* gu,const double* dn,int* own,double bw){
 vector<int> tmp(n),rank(nc);std::iota(rank.begin(),rank.end(),0);
 std::stable_sort(rank.begin(),rank.end(),[&](int a,int b){return gu[a]+dn[a]<gu[b]+dn[b];});
 double best=1e100;
 auto accept=[&](){double val=evaluate6(p,n,nc,ns,gu,dn,tmp.data(),bw);if(val<best){best=val;std::copy(tmp.begin(),tmp.end(),own);}};
 for(int style=0;style<2;style++){solve(p,n,nc,tmp.data(),style,false);accept();}
 vector<int> order(n-ns);std::iota(order.begin(),order.end(),0);vector<double> key(n-ns);
 for(int i=0;i<n-ns;i++){key[i]=1e100;for(int c=0;c<nc;c++)key[i]=std::min(key[i],p[i*nc+c]);}
 std::stable_sort(order.begin(),order.end(),[&](int a,int b){return key[a]>key[b];});
 vector<int> ks={1,2,4,8,16,std::min(nc,ns)};std::sort(ks.begin(),ks.end());ks.erase(std::unique(ks.begin(),ks.end()),ks.end());
 for(int k:ks){if(k>nc||k>ns)continue;
  vector<int> sc(nc,0);
  for(int j=0;j<ns;j++){int c=rank[0];for(int q=1;q<k;q++){int d=rank[q];if((sc[d]+1)*(gu[d]+dn[d])<(sc[c]+1)*(gu[c]+dn[c]))c=d;}sc[c]++;tmp[n-ns+j]=c;}
  double bc=cohort6(sc,gu,dn,bw);vector<double> loads(nc,0);
  for(int c=0;c<nc;c++)if(sc[c])loads[c]=bc;
  for(int i:order){int c=0;for(int j=1;j<nc;j++)if(loads[j]+p[i*nc+j]<loads[c]+p[i*nc+c])c=j;tmp[i]=c;loads[c]+=p[i*nc+c];}
  accept();
 }
 return best;
}
extern "C" void schedule6(const double* p,int n,int nc,int ns,const double* gu,const double* dn,const int* types,double bw,double* bounds,int* own){
 bounds[0]=lower(p,n,nc,types);bounds[1]=upper6(p,n,nc,ns,gu,dn,own,bw);
}
extern "C" void grid6(const double* costs,const double* gu,const double* dn,int nw,int nt,int nshapes,const int* counts,const int* shared,
                     const int* cfg,int ncfg,int maxcore,double bw,double* lbs,double* ubs){
 #pragma omp parallel for schedule(dynamic,8)
 for(int d=0;d<ncfg;d++){
  vector<double> p(nt*maxcore),g(maxcore),v(maxcore);vector<int> own(nt),types(maxcore);
  int nc=0;while(nc<maxcore&&cfg[d*maxcore+nc]>=0)nc++;
  for(int c=0;c<nc;c++)types[c]=cfg[d*maxcore+c];
  for(int w=0;w<nw;w++){
   int n=counts[w];for(int c=0;c<nc;c++){g[c]=gu[types[c]*nw+w];v[c]=dn[types[c]*nw+w];}
   for(int i=0;i<n;i++)for(int c=0;c<nc;c++)p[i*nc+c]=costs[(types[c]*nw+w)*nt+i];
   lbs[d*nw+w]=lower(p.data(),n,nc,types.data());
   ubs[d*nw+w]=upper6(p.data(),n,nc,shared[w],g.data(),v.data(),own.data(),bw);
  }
 }
}
