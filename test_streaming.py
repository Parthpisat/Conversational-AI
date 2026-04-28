#!/usr/bin/env python3
"""Test script to verify backend streaming works correctly."""

import requests
import json
import sys

def test_streaming():
    """Test the /query endpoint and print streaming events."""
    url = "http://localhost:8000/query"
    payload = {
        "question": "Top 10 most ordered products",
        "conversation_history": []
    }
    
    print("🚀 Starting streaming test...")
    print(f"   URL: {url}")
    print(f"   Question: {payload['question']}\n")
    
    try:
        # Make request with streaming enabled
        response = requests.post(url, json=payload, stream=True, timeout=180)
        response.raise_for_status()
        
        print("✅ Connection established! Receiving events...\n")
        print("-" * 80)
        
        event_count = 0
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            
            if line_str.startswith('event: '):
                event_name = line_str.replace('event: ', '')
                print(f"\n📡 Event: {event_name}")
                event_count += 1
            
            elif line_str.startswith('data: '):
                data_str = line_str.replace('data: ', '')
                try:
                    data = json.loads(data_str)
                    
                    # Pretty print the event data
                    print(f"   Step: {data.get('step_number', 'N/A')}")
                    print(f"   Name: {data.get('name', 'N/A')}")
                    print(f"   Status: {data.get('status', 'N/A')}")
                    
                    if data.get('duration_ms'):
                        print(f"   Duration: {data['duration_ms']}ms")
                    
                    if data.get('details'):
                        print(f"   Details: {json.dumps(data['details'], indent=6)}")
                    
                    if data.get('error'):
                        print(f"   ❌ Error: {data['error']}")
                    
                except json.JSONDecodeError as e:
                    print(f"   Data: {data_str[:100]}")
        
        print("\n" + "-" * 80)
        print(f"\n✅ Stream completed! Received {event_count} events")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed! Backend server not running at localhost:8000")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Request timeout! Server took too long to respond")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_streaming()
