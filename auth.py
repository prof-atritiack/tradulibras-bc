"""
Sistema de Autenticação para TraduLibras
Autor: TraduLibras Team
Versão: 2.0.0
"""

import os
import json
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    """Classe de usuário para autenticação"""
    
    def __init__(self, user_id, username, password_hash, role='user', created_at=None):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.role = role  # 'user' ou 'admin'
        self.created_at = created_at or datetime.now().isoformat()
        self.last_login = None
    
    def check_password(self, password):
        """Verifica se a senha está correta"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Verifica se o usuário é admin"""
        return self.role == 'admin'
    
    def to_dict(self):
        """Converte usuário para dicionário"""
        return {
            'id': self.id,
            'username': self.username,
            'password_hash': self.password_hash,
            'role': self.role,
            'created_at': self.created_at,
            'last_login': self.last_login
        }
    
    @classmethod
    def from_dict(cls, data):
        """Cria usuário a partir de dicionário"""
        user = cls(
            user_id=data['id'],
            username=data['username'],
            password_hash=data['password_hash'],
            role=data.get('role', 'user'),
            created_at=data.get('created_at')
        )
        user.last_login = data.get('last_login')
        return user

class UserManager:
    """Gerenciador de usuários"""
    
    def __init__(self, users_file='users.json'):
        self.users_file = users_file
        self.users = {}
        self.load_users()
    
    def load_users(self):
        """Carrega usuários do arquivo"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_data in data.get('users', []):
                        user = User.from_dict(user_data)
                        self.users[user.id] = user
            except Exception as e:
                print(f"Erro ao carregar usuários: {e}")
                self.create_default_users()
        else:
            self.create_default_users()
    
    def save_users(self):
        """Salva usuários no arquivo"""
        try:
            data = {
                'users': [user.to_dict() for user in self.users.values()],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar usuários: {e}")
    
    def create_default_users(self):
        """Cria usuários padrão"""
        # Usuário admin padrão
        admin_user = User(
            user_id='admin',
            username='admin',
            password_hash=generate_password_hash('admin123'),
            role='admin'
        )
        
        # Usuário comum padrão
        regular_user = User(
            user_id='user',
            username='user',
            password_hash=generate_password_hash('user123'),
            role='user'
        )
        
        self.users['admin'] = admin_user
        self.users['user'] = regular_user
        self.save_users()
        
        print("Usuários padrão criados:")
        print("Admin: admin / admin123")
        print("Usuário: user / user123")
    
    def get_user(self, user_id):
        """Obtém usuário por ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username):
        """Obtém usuário por nome de usuário"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def authenticate(self, username, password):
        """Autentica usuário"""
        user = self.get_user_by_username(username)
        if user and user.check_password(password):
            user.last_login = datetime.now().isoformat()
            self.save_users()
            return user
        return None
    
    def create_user(self, username, password, role='user'):
        """Cria novo usuário"""
        if self.get_user_by_username(username):
            return None  # Usuário já existe
        
        user_id = username.lower().replace(' ', '_')
        user = User(
            user_id=user_id,
            username=username,
            password_hash=generate_password_hash(password),
            role=role
        )
        
        self.users[user_id] = user
        self.save_users()
        return user
    
    def update_user(self, user_id, **kwargs):
        """Atualiza usuário"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        for key, value in kwargs.items():
            if hasattr(user, key) and key != 'id':
                setattr(user, key, value)
        
        self.save_users()
        return True
    
    def delete_user(self, user_id):
        """Remove usuário"""
        if user_id in self.users:
            del self.users[user_id]
            self.save_users()
            return True
        return False
    
    def list_users(self):
        """Lista todos os usuários"""
        return list(self.users.values())
    
    def get_stats(self):
        """Obtém estatísticas dos usuários"""
        total_users = len(self.users)
        admin_count = sum(1 for user in self.users.values() if user.is_admin())
        user_count = total_users - admin_count
        
        return {
            'total_users': total_users,
            'admin_count': admin_count,
            'user_count': user_count,
            'last_updated': datetime.now().isoformat()
        }

# Instância global do gerenciador de usuários
user_manager = UserManager()


