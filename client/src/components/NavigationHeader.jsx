import React from 'react'
import { SidebarTrigger } from '@/components/ui/sidebar'
import { NavigationMenu, NavigationMenuContent, NavigationMenuItem, NavigationMenuList, NavigationMenuTrigger } from '@/components/ui/navigation-menu'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Bell, HelpCircle, Search } from 'lucide-react'
import { ThemeSwitcher } from './ThemeSwitcher'

export function NavigationHeader({ user, onLogout, onGoToHome }) {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <SidebarTrigger />
          
          <NavigationMenu>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuTrigger>
                  Blood Tests
                </NavigationMenuTrigger>
                <NavigationMenuContent>
                  <div className="grid w-80 gap-3 p-4">
                    <div className="grid gap-2">
                      <h4 className="font-medium">Blood Analysis</h4>
                      <p className="text-sm text-muted-foreground">
                        CBC, Hemoglobin, Platelet count, Blood sugar analysis
                      </p>
                    </div>
                    <div className="grid gap-2">
                      <h4 className="font-medium">Infectious Disease</h4>
                      <p className="text-sm text-muted-foreground">
                        HIV, HBsAg, Blood group, Rh factor testing
                      </p>
                    </div>
                  </div>
                </NavigationMenuContent>
              </NavigationMenuItem>

              <NavigationMenuItem>
                <NavigationMenuTrigger>
                  Tissue Analysis
                </NavigationMenuTrigger>
                <NavigationMenuContent>
                  <div className="grid w-80 gap-3 p-4">
                    <div className="grid gap-2">
                      <h4 className="font-medium">Biopsy Analysis</h4>
                      <p className="text-sm text-muted-foreground">
                        Cancer detection in tissue samples using H&E staining
                      </p>
                    </div>
                    <div className="grid gap-2">
                      <h4 className="font-medium">WSI Processing</h4>
                      <p className="text-sm text-muted-foreground">
                        Whole slide image analysis with multi-scale attention
                      </p>
                    </div>
                  </div>
                </NavigationMenuContent>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="hidden md:flex">
            {user?.role ? `${user.role} Mode` : 'System Status: Online'}
          </Badge>
          
          <ThemeSwitcher />
          
          <Button variant="ghost" size="sm">
            <Search className="h-4 w-4" />
          </Button>
          
          <Button variant="ghost" size="sm">
            <Bell className="h-4 w-4" />
            <Badge className="ml-1 h-5 w-5 rounded-full p-0 text-xs">3</Badge>
          </Button>
          
          <Button variant="ghost" size="sm">
            <HelpCircle className="h-4 w-4" />
          </Button>

          {onGoToHome && (
            <Button variant="outline" size="sm" onClick={onGoToHome}>
              Home
            </Button>
          )}
        </div>
      </div>
    </header>
  )
}