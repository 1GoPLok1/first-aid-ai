import asyncio
import aiohttp
import random
import json
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RequestEmulator:

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"emulator_{datetime.now().timestamp()}"
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0
        }
    
    # Базы тестовых запросов
    PMP_QUERIES = [
        "Как правильно делать искусственное дыхание?",
        "Что делать при сильном кровотечении из раны?",
        "Как наложить жгут при артериальном кровотечении?",
        "Первая помощь при ожоге второй степени",
        "Как распознать инсульт и что делать?",
        "Что делать если человек подавился?",
        "Как оказать помощь при переломе руки?",
        "Первая помощь при тепловом ударе",
        "Что делать при эпилептическом припадке?",
        "Как обработать глубокую рану?"
    ]
    
    ZOLH_QUERIES = [
        "Сколько воды нужно пить в день взрослому?",
        "Какая суточная норма белка для спортсмена?",
        "Сколько калорий нужно употреблять в день?",
        "Как правильно начать бегать по утрам?",
        "Что такое здоровое питание?",
        "Сколько часов нужно спать для здоровья?",
        "Как снизить уровень стресса?",
        "Какие витамины нужны зимой?",
        "Как правильно делать зарядку?",
        "Что такое интервальное голодание?"
    ]
    
    async def emulate_single_request(self, category: str = None) -> Dict:
        """Эмуляция одного пользовательского запроса"""
        
        # Выбор категории и запроса
        if category == "pmp":
            query = random.choice(self.PMP_QUERIES)
        elif category == "zolh":
            query = random.choice(self.ZOLH_QUERIES)
        else:
            query = random.choice(self.PMP_QUERIES + self.ZOLH_QUERIES)
            category = "pmp" if query in self.PMP_QUERIES else "zolh"
        
        # Формирование payload
        payload = {
            "query": query,
            "category": category,
            "session_id": self.session_id,
            "top_k": random.randint(3, 7)
        }
        
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/query",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    self.stats["total_requests"] += 1
                    self.stats["total_time"] += elapsed
                    
                    if response.status == 200:
                        self.stats["successful"] += 1
                        result = await response.json()
                        return {
                            "status": "success",
                            "query": query,
                            "category": category,
                            "time_ms": round(elapsed * 1000, 2),
                            "results_count": len(result.get("context_fragments", [])),
                            "confidence": result.get("confidence_score", 0)
                        }
                    else:
                        self.stats["failed"] += 1
                        return {
                            "status": "error",
                            "query": query,
                            "http_status": response.status,
                            "time_ms": round(elapsed * 1000, 2)
                        }
                        
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Request failed: {e}")
            return {
                "status": "exception",
                "query": query,
                "error": str(e)
            }
    
    async def emulate_user_session(self, num_requests: int = 10, 
                                   delay_range: tuple = (2, 8)):
        """
        Эмуляция сессии пользователя с несколькими запросами
        и реалистичными задержками между ними.
        """
        logger.info(f"Starting user session emulation: {num_requests} requests")
        
        results = []
        for i in range(num_requests):
            result = await self.emulate_single_request()
            results.append(result)
            
            # Логирование прогресса
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {i+1}/{num_requests} requests completed")
            
            # Реалистичная задержка (как будто пользователь читает ответ)
            if i < num_requests - 1:
                delay = random.uniform(*delay_range)
                await asyncio.sleep(delay)
        
        return results
    
    async def run_load_test(self, concurrent_users: int = 10,
                            requests_per_user: int = 5):
        """
        Нагрузочное тестирование: эмуляция нескольких 
        одновременных пользователей.
        """
        logger.info(f"Starting load test: {concurrent_users} concurrent users")
        
        tasks = [
            self.emulate_user_session(requests_per_user)
            for _ in range(concurrent_users)
        ]
        
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Агрегация статистики
        total_requests = sum(
            1 for user_result in results 
            if isinstance(user_result, list) 
            for _ in user_result
        )
        
        print("\n" + "="*60)
        print("LOAD TEST RESULTS")
        print("="*60)
        print(f"Concurrent users: {concurrent_users}")
        print(f"Requests per user: {requests_per_user}")
        print(f"Total requests: {total_requests}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Requests/second: {total_requests/total_time:.2f}")
        print(f"Success rate: {self.stats['successful']/self.stats['total_requests']*100:.1f}%")
        print(f"Average response time: {self.stats['total_time']/self.stats['total_requests']*1000:.2f} ms")
        print("="*60)
        
        return results

# Использование эмулятора
async def main():
    emulator = RequestEmulator()
    
    # Тест 1: Одиночная сессия
    print("Test 1: Single user session")
    session_results = await emulator.emulate_user_session(num_requests=10)
    
    # Тест 2: Нагрузочное тестирование
    print("\nTest 2: Load test")
    await emulator.run_load_test(concurrent_users=5, requests_per_user=3)

if __name__ == "__main__":
    asyncio.run(main())