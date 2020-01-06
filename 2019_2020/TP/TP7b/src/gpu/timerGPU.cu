#include "timerGPU.h"
#include <cuda.h>
#include <string>

struct _TimerGPU {
  cudaEvent_t m_startEvent, m_stopEvent;
};
  
TimerGPU::TimerGPU(const char *s) : m_running(false), m_elapsed(0.0), m_t(new _TimerGPU) {
  cudaEventCreate(&(m_t->m_startEvent));
  cudaEventCreate(&(m_t->m_stopEvent));
}

TimerGPU::~TimerGPU() {
  cudaEventDestroy(m_t->m_startEvent);
  cudaEventDestroy(m_t->m_stopEvent);
  delete m_t;
}

void TimerGPU::start() {
  if (not m_running) {
    cudaEventRecord(m_t->m_startEvent,0);
    m_running = true;
  }
}

void TimerGPU::stop() {
  if (m_running) {
    float ms;
    
    cudaEventRecord(m_t->m_stopEvent,0);
    cudaEventSynchronize(m_t->m_stopEvent);
    cudaEventElapsedTime(&ms, m_t->m_startEvent, m_t->m_stopEvent);
    
    m_elapsed += ms;
    m_running = false;
  }
}
  
double TimerGPU::elapsed() { return m_elapsed * 0.001; }
  
