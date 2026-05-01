import os, sys

##Функция запуска скрипта
def run_script(script_path):
    """Запустить указанный скрипт Python."""
    try:
        os.system(f'python {script_path}')
    except Exception as e:
        print(f"Произошла ошибка при запуске скрипта {script_path}: {e}")

##Меню services
def services_menu():
    while True:
        print("Scripts menu")
        print("1. Create snapshot")
        print("2. Restore DB from snapshot")
        print("3. Update DB schema")
        print("4. Back")

        choise = input("Enter the number of option: ")

        if choise == 1:
            run_script('backend/scripts/services/create_snapshot.py')
        elif choise == 2:
            run_script('backend/scripts/services/restore_from_snapshot.py')
        elif choise == 3:
            run_script('backend/scripts/services/update_schema.py')
        elif choise == 4:
            break
        else:
            print("Wrong option! Try other")
##Меню scripts
def scripts_menu():
    while True:
        print("Scripts menu")
        print("1. Data")
        print("2. Services")
        print("3. Back")

        choise = input("Enter the number of option: ")

        if choise == 1:
            print("Cannot to enter menu! Menu is in development")
        elif choise == 2:
            services_menu()
        elif choise == 3:
            break
        else:
            print("Wrong option! Try other")
##Меню tests
def tests_menu():
    while True:
        print("Tests menu")
        print("1. Test connection")
        print("2. Test perfomance")
        print("3. Test integrations")
        print("4. Test setup")
        print("5. Back")

        choise = input("Enter the number of option: ")

        if choise == 1:
            run_script('backend/tests/test_conection.py')
        elif choise == 2:
            run_script('backend/tests/test_performance.py')
        elif choise == 3:
            run_script('backend/tests/test_integrations.py')
        elif choise == 4:
            run_script('backend/tests/test_setup.py')
        elif choise == 5:
            break
        else:
            print("Wrong option! Try other")

##Основное меню
def main_menu():
    while True:
        print("Developer app first_aid_ai")
        print("1. Scripts")
        print("2. Tests")
        print("3. Exit")

        choise = input("Enter the number of option: ")

        if  choise == 1:
            scripts_menu()
        elif  choise == 2:
            tests_menu()
        elif  choise == 3:
            print("Exit the app.")
            sys.exit(0)
        else:
            print("Wrong input! Try other")

if __name__ == "__main__":
    main_menu()
        