from enum import IntEnum, StrEnum


class NodeType(StrEnum):
    """Типы узлов в гибридной сети."""
    fiber = "fiber"      
    gen5 = "5G"          
    satellite = "satellite"  
    microwave = "microwave"  
    starlink = "starlink"    
    hybrid = "hybrid"    


class TrafficPriority(IntEnum):
    """Приоритеты для различных типов трафика."""
    CRITICAL = 1    
    HIGH = 2        
    MEDIUM = 3      
    LOW = 4         
    BACKGROUND = 5  


class NetworkStatus(StrEnum):
    """Статусы состояния сети."""
    OPTIMAL = "optimal"           
    DEGRADED = "degraded"         
    CRITICAL = "critical"         
    RECOVERING = "recovering"     
    MAINTENANCE = "maintenance"   


class TrafficType(StrEnum):
    """Типы трафика в сети."""
    VOICE = "voice"                
    VIDEO = "video"                
    INTERACTIVE = "interactive"    
    STREAMING = "streaming"        
    DATA = "data"                  
    IOT = "iot"                    
    SIGNALING = "signaling"        
