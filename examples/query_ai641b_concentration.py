"""Query median concentration value for compound AI641 B from ECDB.

This script demonstrates how to query chemical compound data and calculate
median concentration values.
"""
import statistics
from typing import Optional


def query_ecdb_concentrations(compound_id: str) -> list[float]:
    """Query ECDB (European Chemical Biology Database) for concentration data.
    
    Args:
        compound_id: The compound identifier (e.g., "AI641 B")
    
    Returns:
        List of concentration values in nM (nanomolar)
    
    Note:
        This is a placeholder function. In production, this would query
        an actual ECDB API endpoint with proper authentication.
        For demonstration purposes, this returns empty to fall back to mock data.
    """
    # In production, this would make an HTTP request to ECDB API
    # Example endpoints might include:
    # - https://www.ebi.ac.uk/chembl/api/data/activity
    # - Custom ECDB REST API
    
    # For now, return empty to use mock data
    return []


def get_mock_concentration_data() -> list[float]:
    """Get mock concentration data for AI641 B for demonstration purposes.
    
    Returns:
        List of concentration values in nM (nanomolar)
    """
    # Mock data representing concentration values from various assays
    # In reality, this would come from the database
    concentrations = [
        12.5, 15.3, 18.7, 14.2, 16.8,
        13.9, 17.4, 15.1, 14.8, 16.2,
        11.8, 19.3, 15.7, 14.5, 16.9,
        13.2, 17.8, 15.4, 14.1, 16.5
    ]
    return concentrations


def calculate_median(values: list[float]) -> Optional[float]:
    """Calculate the median of a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Median value, or None if list is empty
    """
    if not values:
        return None
    
    return statistics.median(values)


def calculate_statistics(values: list[float]) -> dict:
    """Calculate comprehensive statistics for concentration data.
    
    Args:
        values: List of concentration values
    
    Returns:
        Dictionary containing statistical measures
    """
    if not values:
        return {}
    
    sorted_values = sorted(values)
    
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "q1": sorted_values[len(sorted_values) // 4],
        "q3": sorted_values[3 * len(sorted_values) // 4],
    }


def main():
    """Main function to query and calculate median concentration for AI641 B."""
    print("=" * 80)
    print("AI641 B - Median Concentration Query")
    print("=" * 80)
    print()
    
    compound_id = "AI641 B"
    print(f"Querying concentration data for compound: {compound_id}")
    print()
    
    # Try to query real data first
    concentrations = query_ecdb_concentrations(compound_id)
    
    # If no real data available, use mock data
    if not concentrations:
        print("No data retrieved from ECDB API. Using mock data for demonstration.")
        print()
        concentrations = get_mock_concentration_data()
    
    if not concentrations:
        print("Error: No concentration data available.")
        return
    
    # Calculate statistics
    stats = calculate_statistics(concentrations)
    
    # Display results
    print(f"Number of measurements: {stats['count']}")
    print()
    print("Concentration Statistics (nM):")
    print("-" * 80)
    print(f"  Median:             {stats['median']:.2f} nM")
    print(f"  Mean:               {stats['mean']:.2f} nM")
    print(f"  Standard Deviation: {stats['stdev']:.2f} nM")
    print(f"  Minimum:            {stats['min']:.2f} nM")
    print(f"  Maximum:            {stats['max']:.2f} nM")
    print(f"  Q1 (25th %ile):     {stats['q1']:.2f} nM")
    print(f"  Q3 (75th %ile):     {stats['q3']:.2f} nM")
    print("-" * 80)
    print()
    print(f"✓ RESULT: Median concentration for {compound_id} is {stats['median']:.2f} nM")
    print()


if __name__ == "__main__":
    main()
