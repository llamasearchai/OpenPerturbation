"""
Performance benchmarks for OpenPerturbation API.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
import httpx
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test utilities
try:
    from src.api.server import create_app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

@pytest.mark.benchmark
@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestAPIPerformance:
    """Test API endpoint performance."""

    @pytest.fixture(scope="class")
    def app(self):
        """Create test app."""
        return create_app()

    @pytest.fixture(scope="class")
    async def client(self, app):
        """Create test client."""
        if app is None:
            pytest.skip("App creation failed")
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_root_endpoint_performance(self, client, benchmark):
        """Benchmark root endpoint response time."""
        async def make_request():
            response = await client.get("/")
            return response
        
        result = benchmark(asyncio.run, make_request())
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_performance(self, client, benchmark):
        """Benchmark health check endpoint."""
        async def make_request():
            response = await client.get("/health")
            return response
        
        result = benchmark(asyncio.run, make_request())
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_models_endpoint_performance(self, client, benchmark):
        """Benchmark models listing endpoint."""
        async def make_request():
            response = await client.get("/models")
            return response
        
        result = benchmark(asyncio.run, make_request())
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_system_info_performance(self, client, benchmark):
        """Benchmark system info endpoint."""
        async def make_request():
            response = await client.get("/system/info")
            return response
        
        result = benchmark(asyncio.run, make_request())
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, benchmark):
        """Benchmark concurrent request handling."""
        async def make_concurrent_requests():
            tasks = []
            for _ in range(10):
                task = client.get("/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            return responses
        
        responses = benchmark(asyncio.run, make_concurrent_requests())
        assert len(responses) == 10
        assert all(r.status_code == 200 for r in responses)

@pytest.mark.benchmark
class TestDataProcessingPerformance:
    """Test data processing performance."""

    def test_small_dataset_processing(self, benchmark):
        """Benchmark processing of small datasets."""
        import numpy as np
        
        def process_small_data():
            # Simulate small dataset processing
            data = np.random.rand(100, 10)
            result = np.mean(data, axis=0)
            return result
        
        result = benchmark(process_small_data)
        assert len(result) == 10

    def test_medium_dataset_processing(self, benchmark):
        """Benchmark processing of medium datasets."""
        import numpy as np
        
        def process_medium_data():
            # Simulate medium dataset processing
            data = np.random.rand(1000, 50)
            # Simulate some computation
            result = np.corrcoef(data.T)
            return result
        
        result = benchmark(process_medium_data)
        assert result.shape == (50, 50)

    def test_large_dataset_processing(self, benchmark):
        """Benchmark processing of large datasets."""
        import numpy as np
        
        def process_large_data():
            # Simulate large dataset processing
            data = np.random.rand(5000, 100)
            # Simulate PCA-like computation
            cov_matrix = np.cov(data.T)
            eigenvals, _ = np.linalg.eigh(cov_matrix)
            return eigenvals
        
        result = benchmark(process_large_data)
        assert len(result) == 100

@pytest.mark.benchmark
class TestAlgorithmPerformance:
    """Test algorithm performance."""

    def test_correlation_computation(self, benchmark):
        """Benchmark correlation computation."""
        import numpy as np
        
        def compute_correlations():
            # Simulate correlation-based causal discovery
            data = np.random.rand(1000, 20)
            corr_matrix = np.corrcoef(data.T)
            # Threshold correlations
            threshold = 0.3
            causal_edges = np.abs(corr_matrix) > threshold
            return causal_edges
        
        result = benchmark(compute_correlations)
        assert result.shape == (20, 20)

    def test_matrix_operations(self, benchmark):
        """Benchmark matrix operations."""
        import numpy as np
        
        def matrix_operations():
            # Simulate heavy matrix computations
            A = np.random.rand(500, 500)
            B = np.random.rand(500, 500)
            C = np.dot(A, B)
            eigenvals = np.linalg.eigvals(C)
            return eigenvals
        
        result = benchmark(matrix_operations)
        assert len(result) == 500

    def test_statistical_computations(self, benchmark):
        """Benchmark statistical computations."""
        import numpy as np
        from scipy import stats
        
        def statistical_tests():
            # Simulate statistical testing
            data1 = np.random.normal(0, 1, 1000)
            data2 = np.random.normal(0.5, 1, 1000)
            
            # T-test
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            # Correlation test
            corr, corr_p = stats.pearsonr(data1[:500], data2[:500])
            
            return t_stat, p_value, corr, corr_p
        
        result = benchmark(statistical_tests)
        assert len(result) == 4

@pytest.mark.benchmark
class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_memory_efficient_processing(self, benchmark):
        """Test memory-efficient data processing."""
        import numpy as np
        
        def memory_efficient_process():
            # Process data in chunks to test memory efficiency
            total_sum = 0
            chunk_size = 1000
            
            for i in range(10):
                chunk = np.random.rand(chunk_size, 100)
                chunk_sum = np.sum(chunk)
                total_sum += chunk_sum
                # Explicitly delete to free memory
                del chunk
            
            return total_sum
        
        result = benchmark(memory_efficient_process)
        assert isinstance(result, (int, float))

    def test_large_array_creation(self, benchmark):
        """Benchmark large array creation and deletion."""
        import numpy as np
        
        def create_large_arrays():
            arrays = []
            for i in range(5):
                arr = np.random.rand(2000, 1000)
                arrays.append(np.mean(arr))  # Store only the mean
            return arrays
        
        result = benchmark(create_large_arrays)
        assert len(result) == 5

@pytest.mark.benchmark
class TestAPIThroughput:
    """Test API throughput under load."""

    @pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
    @pytest.mark.asyncio
    async def test_request_throughput(self, benchmark):
        """Test request throughput."""
        app = create_app()
        if app is None:
            pytest.skip("App creation failed")
        
        async def throughput_test():
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                start_time = time.time()
                tasks = []
                
                # Create 50 concurrent requests
                for _ in range(50):
                    task = client.get("/health")
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks)
                end_time = time.time()
                
                successful_requests = sum(1 for r in responses if r.status_code == 200)
                duration = end_time - start_time
                throughput = successful_requests / duration
                
                return throughput, successful_requests, duration
        
        throughput, successful, duration = benchmark(asyncio.run, throughput_test())
        
        # Assert reasonable performance
        assert successful == 50  # All requests should succeed
        assert throughput > 10  # At least 10 requests per second
        assert duration < 10  # Should complete within 10 seconds

if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "--benchmark-only"]) 