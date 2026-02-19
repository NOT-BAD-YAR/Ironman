"""
GPS Helper for Raspberry Pi 5
Gets real-time GPS coordinates for accurate pothole location reporting
"""

import time
import threading

try:
    import gps
    GPS_LIBRARY_AVAILABLE = True
except ImportError:
    GPS_LIBRARY_AVAILABLE = False
    print("⚠️  python3-gps not installed. Install with: sudo apt install python3-gps")


class GPSHelper:
    """
    GPS helper class for getting real-time coordinates
    """
    
    def __init__(self):
        self.session = None
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.running = False
        self.thread = None
        self.last_update = 0
        self.fix_quality = 0  # 0 = no fix, 1 = GPS fix, 2 = DGPS fix
        
    def start(self):
        """Start GPS monitoring"""
        if not GPS_LIBRARY_AVAILABLE:
            print("⚠️  GPS library not available")
            return False
            
        try:
            self.session = gps.gps(mode=gps.WATCH_ENABLE)
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print("✅ GPS started, waiting for fix...")
            return True
        except Exception as e:
            print(f"⚠️  GPS initialization failed: {e}")
            print("   Make sure GPS daemon is running: sudo systemctl start gpsd")
            self.running = False
            return False
    
    def _update_loop(self):
        """Background thread to update GPS coordinates"""
        while self.running:
            try:
                report = self.session.next()
                
                if report['class'] == 'TPV':
                    if hasattr(report, 'lat') and hasattr(report, 'lon'):
                        if report.lat and report.lon:
                            self.latitude = report.lat
                            self.longitude = report.lon
                            self.last_update = time.time()
                            
                            if hasattr(report, 'alt'):
                                self.altitude = report.alt
                            
                            # Check fix quality
                            if hasattr(report, 'mode'):
                                self.fix_quality = report.mode
                                
            except StopIteration:
                break
            except Exception as e:
                time.sleep(1)
    
    def get_coordinates(self):
        """
        Get current GPS coordinates
        
        Returns:
            tuple: (latitude, longitude) or (None, None) if no fix
        """
        # Check if GPS data is recent (within last 10 seconds)
        if self.latitude and self.longitude:
            if time.time() - self.last_update < 10:
                return self.latitude, self.longitude
        
        return None, None
    
    def has_fix(self):
        """Check if GPS has a fix"""
        lat, lng = self.get_coordinates()
        return lat is not None and lng is not None
    
    def get_status(self):
        """Get GPS status information"""
        lat, lng = self.get_coordinates()
        return {
            'has_fix': self.has_fix(),
            'latitude': lat,
            'longitude': lng,
            'altitude': self.altitude,
            'fix_quality': self.fix_quality,
            'last_update': self.last_update
        }
    
    def stop(self):
        """Stop GPS monitoring"""
        self.running = False
        if self.session:
            try:
                self.session.close()
            except:
                pass
        print("GPS stopped")


def get_gps_coordinates_simple():
    """
    Simple function to get GPS coordinates (for quick use)
    
    Returns:
        tuple: (latitude, longitude) or (None, None)
    """
    try:
        helper = GPSHelper()
        if helper.start():
            time.sleep(3)  # Wait for GPS fix
            return helper.get_coordinates()
    except:
        pass
    return None, None


