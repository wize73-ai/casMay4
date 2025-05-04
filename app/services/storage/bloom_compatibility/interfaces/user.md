# UserManager Interface

The `UserManager` class provides an interface for managing user data, profiles, sessions, and language preferences.

## Methods

### create_user(user_data)

Creates a new user in the system.

**Parameters:**
- `user_data` (object): User information including:
  - `username` (string): Required username
  - `email` (string): Optional email
  - `password_hash` (string): Optional password hash
  - `settings` (object): Optional user settings
  - `profile` (object): Optional user profile
  - `languages` (array): Optional list of language objects

**Returns:**
- `user_id` (string): Unique identifier for the created user

**TypeScript Equivalent:**
```typescript
async createUser(userData: UserData): Promise<string>