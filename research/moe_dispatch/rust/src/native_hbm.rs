//! Opt-in pinned Ramulator CAPI v2 backend. Core edge = 1 ns; sectors = 32 B.
use serde_json::{Value,json};
use std::{cell::RefCell,collections::VecDeque,ffi::{CString,CStr,c_char,c_void},rc::Rc};
#[link(name="ramulator")]
unsafe extern "C" {
    fn ramulator_new(config:*const c_char)->*mut c_void;
    fn ramulator_finalize(raw:*mut c_void);
    fn ramulator_request(raw:*mut c_void,address:u64,write:bool,cb:extern "C" fn(*mut c_void),data:*mut c_void,size:i32)->bool;
    fn ramulator_period(raw:*mut c_void)->f32;
    fn ramulator_tick(raw:*mut c_void);
    fn ramulator_capi_version()->u32;
    fn ramulator_tx_bytes(raw:*mut c_void)->u32;
    fn ramulator_stats(raw:*mut c_void,buffer:*mut c_char,capacity:u64)->u64;
    fn ramulator_library_path()->*const c_char;
}
struct Callback { serial:u64, queue:Rc<RefCell<VecDeque<u64>>> }
extern "C" fn returned(ptr:*mut c_void) {
    let cb=unsafe { Box::from_raw(ptr.cast::<Callback>()) };
    cb.queue.borrow_mut().push_back(cb.serial);
}
pub(super) struct Backend { raw:*mut c_void,queue:Rc<RefCell<VecDeque<u64>>>,tick:u64,
    pub pending:usize,pub accepted:u64,pub completed:u64,pub rejected:u64 }
impl Backend {
    pub fn new(config:&Value)->Self {
        unsafe { assert_eq!(ramulator_capi_version(),2,"wrong native CAPI ABI"); }
        let config=CString::new(serde_json::to_string(config).unwrap()).unwrap();
        let raw=unsafe { ramulator_new(config.as_ptr()) };
        assert!(!raw.is_null(),"native HBM initialization failed");
        assert_eq!(unsafe{ramulator_tx_bytes(raw)},32,"native transaction must equal DMA 32B sector");
        assert!((unsafe{ramulator_period(raw)}-1.0).abs()<1e-6,"campaign requires native 1ns period");
        Self { raw,queue:Rc::new(RefCell::new(VecDeque::new())),tick:0,pending:0,accepted:0,completed:0,rejected:0 }
    }
    pub fn request(&mut self,serial:u64,address:u64)->bool {
        assert_eq!(address%32,0);
        let cb=Box::into_raw(Box::new(Callback {serial,queue:Rc::clone(&self.queue)}));
        let accepted=unsafe {ramulator_request(self.raw,address,false,returned,cb.cast(),32)};
        if accepted { self.pending+=1;self.accepted+=1; }
        else { unsafe {drop(Box::from_raw(cb));} self.rejected+=1; }
        accepted
    }
    pub fn advance(&mut self,now:u64)->Vec<u64> {
        assert!(now>=self.tick);
        while self.tick<now { unsafe {ramulator_tick(self.raw)} self.tick+=1; }
        let out:Vec<_>=self.queue.borrow_mut().drain(..).collect();
        self.pending-=out.len();self.completed+=out.len() as u64;out
    }
    pub fn report(&self)->Value {
        assert_eq!(self.pending,0);assert!(self.queue.borrow().is_empty());assert_eq!(self.accepted,self.completed);
        let required=unsafe{ramulator_stats(self.raw,std::ptr::null_mut(),0)};
        let mut buffer=vec![0u8;required as usize];
        assert_eq!(unsafe{ramulator_stats(self.raw,buffer.as_mut_ptr().cast(),required)},required);
        let stats=unsafe{CStr::from_ptr(buffer.as_ptr().cast())}.to_str().unwrap();
        let path=unsafe{ramulator_library_path()};assert!(!path.is_null());
        json!({"backend":"Ramulator2 CAPI v2","library_path":unsafe{CStr::from_ptr(path)}.to_str().unwrap(),
            "native_tick_ns":1,"transaction_bytes":32,"ticks":self.tick,"accepted":self.accepted,
            "completed":self.completed,"rejected_attempts":self.rejected,"pending":self.pending,
            "native_stats_yaml":stats})
    }
}
impl Drop for Backend { fn drop(&mut self) { unsafe{ramulator_finalize(self.raw)} } }
