from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class UserInfo(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime]
    status: str