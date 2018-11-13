#include <chrono>

class Timer {
	public:
	    Timer() { m_elapsed = 0.0; }
	    void init() { m_elapsed = 0.0; }
		void start();
		void stop();
		double elapsed() { return m_elapsed; }
	protected:
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_end;
        double m_elapsed;
};
