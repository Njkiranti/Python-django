from django.db import models

# Create your models here.
class Gender(models.Model):
    gender = models.CharField(max_length=100)

    def __str__(self):
        return self.gender
    
class Position(models.Model):
    position = models.CharField(max_length=100)

    def __str__(self):
        return self.position
    
class IsActive(models.Model):
    isactive = models.CharField(max_length=100)

    def __str__(self):
        return self.isactive

class File_hash_value(models.Model):
    file_name = models.CharField(max_length=100)
    hash_value = models.CharField(max_length=100)
    obfuscatedvalue = models.CharField(max_length=100)
    deobfuscatdvalue = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

class Staff(models.Model):
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    firstname = models.CharField(max_length=100)
    middlename = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    mobileno = models.CharField(max_length=100)
    gender = models.ForeignKey(Gender,on_delete = models.CASCADE)
    email = models.CharField(max_length=100)
    position = models.ForeignKey(Position,on_delete = models.CASCADE)
    hash_value = models.ForeignKey(File_hash_value, on_delete= models.CASCADE)
    isactive = models.ForeignKey(IsActive, on_delete= models.CASCADE)


class Data(models.Model):
    name = models.CharField(max_length=100)
    value = models.IntegerField()

class DeviceLocation(models.Model):
    device = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)