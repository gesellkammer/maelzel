import rtmidi2
import time


class MidiNotifier:
	def __init__(self, check_interval=1):
		self.callbacks = []
		self.running = False
		self.check_interval = check_interval
		
	def _loop(self):
		get_in_ports = rtmidi2.get_in_ports
		get_out_ports = rtmidi2.get_out_ports
		check_interval = self.check_interval
		outports = get_out_ports()
		inports = get_in_ports()
		self.running = True
		while self.running:
			inports_now = set(get_in_ports())
			outports_now = set(get_out_ports())
			if inports_now != inports or outports_now != outports:
				inports = inports_now
				outports = outports_now 
				print("ports changed", inports, outports)
				for f in self.callbacks:
					f(inports, outports)
			time.sleep(check_interval)

	def register(self, callback):
		self.callbacks.append(callback)

	def run_in_thread(self):
		import threading
		t = threading.Thread(target=self._loop)
		t.start()
		return t

	def run_blocking(self):
		self._loop()

	def run_async(self, loop=None):
		pass

	def stop(self):
		self.running = False