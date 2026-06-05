import sys
import subprocess
import re
from pathlib import Path
from importlib.metadata import distributions

def get_installed_packages():
    """Возвращает словарь установленных пакетов {имя: версия}"""
    packages = {}
    for dist in distributions():
        packages[dist.metadata["Name"].lower()] = dist.version
    return packages

def parse_package_name(package_spec):
    package_spec = package_spec.strip()
    # Берём всё до первого символа [ > < = ! 
    # Но нужно учесть, что внутри [ ] может быть extra
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)', package_spec)
    if match:
        return match.group(1).lower()
    return package_spec.lower()

def check_requirements(requirements_file='requirements.txt'):
    """Проверяет установлены ли все библиотеки из requirements.txt"""
    req_path = Path(requirements_file)
    if not req_path.exists():
        print(f"❌ Файл {requirements_file} не найден!")
        return False, []
    
    with open(req_path, 'r') as f:
        required_packages = [line.strip() for line in f 
                           if line.strip() and not line.startswith('#')]
    
    installed_packages = get_installed_packages()
    
    missing_packages = []
    for package in required_packages:
        pkg_name = parse_package_name(package)
        
        if pkg_name not in installed_packages:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def install_missing_packages(missing_packages, upgrade=False):
    """Устанавливает недостающие пакеты"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Устанавливаю недостающие пакеты ({len(missing_packages)} шт.):")
    for package in missing_packages:
        print(f"  - {package}")
    
    answer = input("\nУстановить эти пакеты? (y/n): ").lower()
    if answer != 'y':
        print("❌ Установка отменена")
        return False
    
    for package in missing_packages:
        print(f"\nУстанавливаю {package}...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        try:
            subprocess.check_call(cmd)
            print(f"✅ {package} установлен")
        except subprocess.CalledProcessError:
            print(f"❌ Ошибка при установке {package}")
            return False
    
    return True

def show_installed_packages():
    """Показывает список установленных пакетов"""
    packages = get_installed_packages()
    print(f"\n📋 Установленные пакеты ({len(packages)} шт.):")
    for name, version in sorted(packages.items()):
        print(f"  {name}=={version}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Проверка установки библиотек из requirements.txt')
    parser.add_argument('-f', '--file', default='requirements.txt', 
                       help='Путь к файлу requirements.txt')
    parser.add_argument('-i', '--install', action='store_true',
                       help='Автоматически установить недостающие пакеты')
    parser.add_argument('-u', '--upgrade', action='store_true',
                       help='Обновить пакеты при установке')
    parser.add_argument('-l', '--list-installed', action='store_true',
                       help='Показать список установленных пакетов')
    
    args = parser.parse_args()
    
    if args.list_installed:
        show_installed_packages()
        return
    
    print(f"🔍 Проверяю файл: {args.file}")
    print("-" * 50)
    
    all_installed, missing = check_requirements(args.file)
    
    if all_installed:
        print("✅ Все библиотеки из requirements.txt установлены!")
    else:
        print(f"❌ Отсутствуют {len(missing)} библиотек:")
        for package in missing:
            print(f"  - {package}")
        
        if args.install:
            print("\n🚀 Автоматическая установка...")
            install_missing_packages(missing, args.upgrade)
        else:
            print(f"\n💡 Для установки недостающих пакетов выполните:")
            print(f"  pip install -r {args.file}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()