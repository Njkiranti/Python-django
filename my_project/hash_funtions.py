import hashlib
from sympy import false, true
from .models import File_hash_value

# Calculate hash value for a file

def compare_hash_with_database(filename):
    user_hash_values = generate_hash(filename)
    stored_hash_values = File_hash_value.objects.get(file_hash=user_hash_values).file_data
    if (user_hash_values == stored_hash_values):
         return true
    else:
     return false
    

# Generate hash value for a file
def generate_hash(contents):
    hash_object = hashlib.sha256(contents.encode())
    hash_value = hash_object.hexdigest()
    return hash_value



            